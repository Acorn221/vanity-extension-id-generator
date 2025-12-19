/**
 * WebGPU Prime Generator
 *
 * Manages WebGPU compute pipeline for generating 1024-bit primes.
 * Uses conservative dispatch sizes to prevent GPU lockup.
 */

import {
	PRIME_BYTES,
	PRIME_POOL_MAGIC,
	PRIME_POOL_VERSION,
	PRIME_WORDS,
	SMALL_PRIMES
} from './constants'
import shaderSource from './primeGenerator.wgsl?raw'

export interface GPUInfo {
	available: boolean
	deviceName: string
	maxWorkgroupSize: number
	maxBufferSize: number
}

export interface GenerationProgress {
	primesFound: number
	targetCount: number
	rate: number // primes per second
	elapsed: number // seconds
	eta: number // seconds remaining
}

export interface GenerationOptions {
	checkpointInterval?: number // Save checkpoint every N primes (default: 1000)
	onCheckpoint?: (data: Uint8Array, count: number) => void
	throttleMs?: number // Delay between dispatches (default: 10ms)
}

export type ProgressCallback = (progress: GenerationProgress) => void

const CHECKPOINT_STORAGE_KEY = 'prime-generator-checkpoint'

export interface Checkpoint {
	primeData: number[] // Stored as regular array for JSON
	count: number
	targetCount: number
	timestamp: number
}

/**
 * Save checkpoint to localStorage
 */
export function saveCheckpoint(
	data: Uint32Array,
	count: number,
	targetCount: number
): void {
	const checkpoint: Checkpoint = {
		primeData: Array.from(data.slice(0, count * PRIME_WORDS)),
		count,
		targetCount,
		timestamp: Date.now()
	}
	try {
		localStorage.setItem(CHECKPOINT_STORAGE_KEY, JSON.stringify(checkpoint))
	} catch {
		// localStorage might be full or unavailable - silently ignore
	}
}

/**
 * Load checkpoint from localStorage
 */
export function loadCheckpoint(): Checkpoint | null {
	try {
		const data = localStorage.getItem(CHECKPOINT_STORAGE_KEY)
		if (data) {
			return JSON.parse(data) as Checkpoint
		}
	} catch {
		// Ignore parse errors
	}
	return null
}

/**
 * Clear checkpoint from localStorage
 */
export function clearCheckpoint(): void {
	localStorage.removeItem(CHECKPOINT_STORAGE_KEY)
}

/**
 * Check if WebGPU is available in the browser
 */
export async function checkWebGPUSupport(): Promise<GPUInfo> {
	if (!navigator.gpu) {
		return {
			available: false,
			deviceName: 'WebGPU not supported',
			maxWorkgroupSize: 0,
			maxBufferSize: 0
		}
	}

	try {
		const adapter = await navigator.gpu.requestAdapter()
		if (!adapter) {
			return {
				available: false,
				deviceName: 'No GPU adapter found',
				maxWorkgroupSize: 0,
				maxBufferSize: 0
			}
		}

		const info = adapter.info
		const device = await adapter.requestDevice()

		const gpuInfo: GPUInfo = {
			available: true,
			deviceName: info.description || info.vendor || 'Unknown GPU',
			maxWorkgroupSize: device.limits.maxComputeWorkgroupSizeX,
			maxBufferSize: device.limits.maxBufferSize
		}

		device.destroy()
		return gpuInfo
	} catch {
		return {
			available: false,
			deviceName: 'Failed to initialize WebGPU',
			maxWorkgroupSize: 0,
			maxBufferSize: 0
		}
	}
}

/**
 * Sleep for a given number of milliseconds
 */
function sleep(ms: number): Promise<void> {
	return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * WebGPU Prime Generator class
 */
export class PrimeGeneratorGPU {
	private device: GPUDevice | null = null
	private pipeline: GPUComputePipeline | null = null
	private bindGroupLayout: GPUBindGroupLayout | null = null
	private running = false
	private stopRequested = false

	// Buffers
	private outputBuffer: GPUBuffer | null = null
	private countBuffer: GPUBuffer | null = null
	private seedsBuffer: GPUBuffer | null = null
	private paramsBuffer: GPUBuffer | null = null
	private smallPrimesBuffer: GPUBuffer | null = null
	private readbackBuffer: GPUBuffer | null = null
	private countReadbackBuffer: GPUBuffer | null = null

	/**
	 * Initialize WebGPU device and compile shader
	 */
	async initialize(): Promise<GPUInfo> {
		if (!navigator.gpu) {
			throw new Error('WebGPU is not supported in this browser')
		}

		const adapter = await navigator.gpu.requestAdapter({
			powerPreference: 'high-performance'
		})
		if (!adapter) {
			throw new Error('No GPU adapter available')
		}

		const info = adapter.info
		this.device = await adapter.requestDevice({
			requiredLimits: {
				maxBufferSize: adapter.limits.maxBufferSize,
				maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
			}
		})

		// Create shader module
		const shaderModule = this.device.createShaderModule({
			label: 'Prime Generator Shader',
			code: shaderSource
		})

		// Check for compilation errors
		const compilationInfo = await shaderModule.getCompilationInfo()
		for (const message of compilationInfo.messages) {
			if (message.type === 'error') {
				throw new Error(`Shader compilation error: ${message.message}`)
			}
		}

		// Create bind group layout
		this.bindGroupLayout = this.device.createBindGroupLayout({
			label: 'Prime Generator Bind Group Layout',
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					buffer: {type: 'storage'}
				},
				{
					binding: 1,
					visibility: GPUShaderStage.COMPUTE,
					buffer: {type: 'storage'}
				},
				{
					binding: 2,
					visibility: GPUShaderStage.COMPUTE,
					buffer: {type: 'read-only-storage'}
				},
				{
					binding: 3,
					visibility: GPUShaderStage.COMPUTE,
					buffer: {type: 'uniform'}
				},
				{
					binding: 4,
					visibility: GPUShaderStage.COMPUTE,
					buffer: {type: 'read-only-storage'}
				}
			]
		})

		// Create pipeline layout
		const pipelineLayout = this.device.createPipelineLayout({
			label: 'Prime Generator Pipeline Layout',
			bindGroupLayouts: [this.bindGroupLayout]
		})

		// Create compute pipeline
		this.pipeline = this.device.createComputePipeline({
			label: 'Prime Generator Pipeline',
			layout: pipelineLayout,
			compute: {
				module: shaderModule,
				entryPoint: 'generate_primes'
			}
		})

		// Create small primes buffer (constant)
		this.smallPrimesBuffer = this.device.createBuffer({
			label: 'Small Primes Buffer',
			size: SMALL_PRIMES.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		})
		this.device.queue.writeBuffer(
			this.smallPrimesBuffer,
			0,
			SMALL_PRIMES.buffer
		)

		return {
			available: true,
			deviceName: info.description || info.vendor || 'Unknown GPU',
			maxWorkgroupSize: this.device.limits.maxComputeWorkgroupSizeX,
			maxBufferSize: this.device.limits.maxBufferSize
		}
	}

	/**
	 * Generate primes with checkpointing support
	 */
	async generate(
		targetCount: number,
		onProgress?: ProgressCallback,
		options?: GenerationOptions
	): Promise<Uint8Array> {
		if (!(this.device && this.pipeline && this.bindGroupLayout)) {
			throw new Error('GPU not initialized. Call initialize() first.')
		}

		if (this.running) {
			throw new Error('Generation already in progress')
		}

		this.running = true
		this.stopRequested = false

		const checkpointInterval = options?.checkpointInterval ?? 1000
		const throttleMs = options?.throttleMs ?? 10
		const onCheckpoint = options?.onCheckpoint

		const startTime = performance.now()
		let lastCheckpointCount = 0

		try {
			// Calculate buffer sizes
			const outputSize = targetCount * PRIME_WORDS * 4 // u32 words
			const maxBufferSize = this.device.limits.maxStorageBufferBindingSize

			if (outputSize > maxBufferSize) {
				throw new Error(
					`Requested ${targetCount} primes requires ${outputSize} bytes, ` +
						`but max buffer size is ${maxBufferSize} bytes. ` +
						`Maximum primes: ${Math.floor(maxBufferSize / (PRIME_WORDS * 4))}`
				)
			}

			// Create output buffer
			this.outputBuffer = this.device.createBuffer({
				label: 'Output Primes Buffer',
				size: outputSize,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			})

			// Create count buffer (atomic counter)
			this.countBuffer = this.device.createBuffer({
				label: 'Prime Count Buffer',
				size: 4, // single u32
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			})

			// Create seeds buffer (2048 random seeds for up to 1024 threads)
			const seeds = new Uint32Array(2048)
			crypto.getRandomValues(seeds)
			this.seedsBuffer = this.device.createBuffer({
				label: 'Seeds Buffer',
				size: seeds.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			})
			this.device.queue.writeBuffer(this.seedsBuffer, 0, seeds.buffer)

			// Create params buffer
			this.paramsBuffer = this.device.createBuffer({
				label: 'Params Buffer',
				size: 16, // vec4<u32>
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			})

			// Create readback buffers
			this.readbackBuffer = this.device.createBuffer({
				label: 'Output Readback Buffer',
				size: outputSize,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			})

			this.countReadbackBuffer = this.device.createBuffer({
				label: 'Count Readback Buffer',
				size: 4,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			})

			// CONSERVATIVE dispatch parameters to prevent GPU lockup
			const workgroupSize = 64 // Must match shader
			const numWorkgroups = 32 // Much smaller! Only 2k threads
			const batchSize = 5 // Only 5 candidates per thread per dispatch

			let primesFound = 0
			let iteration = 0

			// Main generation loop
			while (primesFound < targetCount && !this.stopRequested) {
				// Update params: [target_count, batch_size, seed_offset, 0]
				const params = new Uint32Array([
					targetCount,
					batchSize,
					iteration * numWorkgroups * workgroupSize,
					0
				])
				this.device.queue.writeBuffer(this.paramsBuffer, 0, params.buffer)

				// Create bind group
				const bindGroup = this.device.createBindGroup({
					label: 'Prime Generator Bind Group',
					layout: this.bindGroupLayout,
					entries: [
						{binding: 0, resource: {buffer: this.outputBuffer}},
						{binding: 1, resource: {buffer: this.countBuffer}},
						{binding: 2, resource: {buffer: this.seedsBuffer}},
						{binding: 3, resource: {buffer: this.paramsBuffer}},
						{binding: 4, resource: {buffer: this.smallPrimesBuffer!}}
					]
				})

				// Create and submit command buffer
				const commandEncoder = this.device.createCommandEncoder()
				const passEncoder = commandEncoder.beginComputePass()
				passEncoder.setPipeline(this.pipeline)
				passEncoder.setBindGroup(0, bindGroup)
				passEncoder.dispatchWorkgroups(numWorkgroups)
				passEncoder.end()

				// Copy count for readback
				commandEncoder.copyBufferToBuffer(
					this.countBuffer,
					0,
					this.countReadbackBuffer,
					0,
					4
				)

				this.device.queue.submit([commandEncoder.finish()])

				// Read back count
				await this.countReadbackBuffer.mapAsync(GPUMapMode.READ)
				const countData = new Uint32Array(
					this.countReadbackBuffer.getMappedRange()
				)
				primesFound = countData[0] ?? 0
				this.countReadbackBuffer.unmap()

				// Report progress
				const elapsed = (performance.now() - startTime) / 1000
				const rate = elapsed > 0 ? primesFound / elapsed : 0
				const eta =
					rate > 0
						? (targetCount - primesFound) / rate
						: Number.POSITIVE_INFINITY

				onProgress?.({
					primesFound,
					targetCount,
					rate,
					elapsed,
					eta
				})

				// Checkpoint if we've generated enough new primes
				if (primesFound - lastCheckpointCount >= checkpointInterval) {
					await this.saveCheckpointData(primesFound, targetCount, onCheckpoint)
					lastCheckpointCount = primesFound
				}

				iteration++

				// Throttle to prevent GPU lockup - give the system breathing room
				if (throttleMs > 0) {
					await sleep(throttleMs)
				}
			}

			// Final readback of all primes
			const finalCount = Math.min(primesFound, targetCount)

			const commandEncoder = this.device.createCommandEncoder()
			commandEncoder.copyBufferToBuffer(
				this.outputBuffer,
				0,
				this.readbackBuffer,
				0,
				finalCount * PRIME_WORDS * 4
			)
			this.device.queue.submit([commandEncoder.finish()])

			await this.readbackBuffer.mapAsync(GPUMapMode.READ)
			const primeData = new Uint32Array(
				this.readbackBuffer.getMappedRange(0, finalCount * PRIME_WORDS * 4)
			)

			// Convert to binary format
			const result = this.createBinaryOutput(primeData, finalCount)

			this.readbackBuffer.unmap()

			// Clear checkpoint on successful completion
			clearCheckpoint()

			return result
		} finally {
			this.cleanup()
			this.running = false
		}
	}

	/**
	 * Save checkpoint data
	 */
	private async saveCheckpointData(
		count: number,
		targetCount: number,
		onCheckpoint?: (data: Uint8Array, count: number) => void
	): Promise<void> {
		if (!(this.outputBuffer && this.readbackBuffer && this.device)) return

		// Read current data
		const tempReadback = this.device.createBuffer({
			label: 'Checkpoint Readback',
			size: count * PRIME_WORDS * 4,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		})

		const encoder = this.device.createCommandEncoder()
		encoder.copyBufferToBuffer(
			this.outputBuffer,
			0,
			tempReadback,
			0,
			count * PRIME_WORDS * 4
		)
		this.device.queue.submit([encoder.finish()])

		await tempReadback.mapAsync(GPUMapMode.READ)
		const data = new Uint32Array(tempReadback.getMappedRange())

		// Save to localStorage
		saveCheckpoint(data, count, targetCount)

		// Call callback if provided
		if (onCheckpoint) {
			const binaryData = this.createBinaryOutput(data, count)
			onCheckpoint(binaryData, count)
		}

		tempReadback.unmap()
		tempReadback.destroy()
	}

	/**
	 * Stop ongoing generation
	 */
	stop(): void {
		this.stopRequested = true
	}

	/**
	 * Check if generation is running
	 */
	isRunning(): boolean {
		return this.running
	}

	/**
	 * Create binary output file in the standard format
	 */
	private createBinaryOutput(
		primeData: Uint32Array,
		count: number
	): Uint8Array {
		// Header: magic (4) + version (4) + count (4) + prime_bytes (4) = 16 bytes
		const headerSize = 16
		const dataSize = count * PRIME_BYTES
		const result = new Uint8Array(headerSize + dataSize)
		const view = new DataView(result.buffer)

		// Write header (little-endian to match C++ structs)
		view.setUint32(0, PRIME_POOL_MAGIC, true)
		view.setUint32(4, PRIME_POOL_VERSION, true)
		view.setUint32(8, count, true)
		view.setUint32(12, PRIME_BYTES, true)

		// Copy prime data (already in big-endian byte order from shader)
		for (let i = 0; i < count; i++) {
			const srcOffset = i * PRIME_WORDS
			const dstOffset = headerSize + i * PRIME_BYTES

			for (let w = 0; w < PRIME_WORDS; w++) {
				const word = primeData[srcOffset + w] ?? 0
				// Write as big-endian bytes
				result[dstOffset + w * 4] = (word >> 24) & 0xff
				result[dstOffset + w * 4 + 1] = (word >> 16) & 0xff
				result[dstOffset + w * 4 + 2] = (word >> 8) & 0xff
				result[dstOffset + w * 4 + 3] = word & 0xff
			}
		}

		return result
	}

	/**
	 * Clean up GPU resources
	 */
	private cleanup(): void {
		this.outputBuffer?.destroy()
		this.countBuffer?.destroy()
		this.seedsBuffer?.destroy()
		this.paramsBuffer?.destroy()
		this.readbackBuffer?.destroy()
		this.countReadbackBuffer?.destroy()

		this.outputBuffer = null
		this.countBuffer = null
		this.seedsBuffer = null
		this.paramsBuffer = null
		this.readbackBuffer = null
		this.countReadbackBuffer = null
	}

	/**
	 * Destroy all resources
	 */
	destroy(): void {
		this.cleanup()
		this.smallPrimesBuffer?.destroy()
		this.smallPrimesBuffer = null
		this.device?.destroy()
		this.device = null
		this.pipeline = null
		this.bindGroupLayout = null
	}
}
