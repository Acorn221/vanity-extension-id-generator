import {useCallback, useEffect, useRef, useState} from 'react'
import {
	checkWebGPUSupport,
	type GenerationProgress,
	type GPUInfo,
	PrimeGeneratorGPU
} from '@/gpu'

export type GeneratorStatus =
	| 'idle'
	| 'initializing'
	| 'ready'
	| 'generating'
	| 'complete'
	| 'error'

export interface LogEntry {
	id: number
	timestamp: Date
	message: string
	type: 'info' | 'success' | 'warning' | 'error'
}

export interface UsePrimeGeneratorResult {
	// Status
	status: GeneratorStatus
	error: string | null
	gpuInfo: GPUInfo | null

	// Progress
	progress: GenerationProgress | null
	lastCheckpoint: number | null

	// Activity log
	logs: LogEntry[]

	// Result
	result: Uint8Array | null

	// Actions
	initialize: () => Promise<void>
	start: (targetCount: number) => Promise<void>
	stop: () => void
	download: (filename?: string) => void
	reset: () => void
	clearLogs: () => void
}

let logIdCounter = 0

/**
 * React hook for WebGPU prime generation
 */
export function usePrimeGenerator(): UsePrimeGeneratorResult {
	const [status, setStatus] = useState<GeneratorStatus>('idle')
	const [error, setError] = useState<string | null>(null)
	const [gpuInfo, setGpuInfo] = useState<GPUInfo | null>(null)
	const [progress, setProgress] = useState<GenerationProgress | null>(null)
	const [result, setResult] = useState<Uint8Array | null>(null)
	const [logs, setLogs] = useState<LogEntry[]>([])
	const [lastCheckpoint, setLastCheckpoint] = useState<number | null>(null)

	const generatorRef = useRef<PrimeGeneratorGPU | null>(null)

	const addLog = useCallback(
		(message: string, type: LogEntry['type'] = 'info') => {
			const entry: LogEntry = {
				id: logIdCounter++,
				timestamp: new Date(),
				message,
				type
			}
			setLogs(prev => [...prev.slice(-49), entry]) // Keep last 50 logs
		},
		[]
	)

	const clearLogs = useCallback(() => {
		setLogs([])
	}, [])

	// Cleanup on unmount
	useEffect(
		() => () => {
			generatorRef.current?.destroy()
		},
		[]
	)

	const initialize = useCallback(async () => {
		if (generatorRef.current) {
			return // Already initialized
		}

		setStatus('initializing')
		setError(null)
		addLog('Checking WebGPU support...')

		try {
			// First check basic support
			const supportInfo = await checkWebGPUSupport()
			if (!supportInfo.available) {
				setError(supportInfo.deviceName)
				setStatus('error')
				addLog(`WebGPU not available: ${supportInfo.deviceName}`, 'error')
				return
			}

			addLog(`Found GPU: ${supportInfo.deviceName}`)
			addLog('Compiling shader...')

			// Initialize generator
			const generator = new PrimeGeneratorGPU()
			const info = await generator.initialize()

			generatorRef.current = generator
			setGpuInfo(info)
			setStatus('ready')
			addLog('Shader compiled successfully', 'success')
			addLog('Ready to generate primes', 'success')
		} catch (err) {
			const message = err instanceof Error ? err.message : 'Unknown error'
			setError(message)
			setStatus('error')
			addLog(`Initialization failed: ${message}`, 'error')
		}
	}, [addLog])

	const start = useCallback(
		async (targetCount: number) => {
			const generator = generatorRef.current
			if (!generator) {
				setError('Generator not initialized')
				addLog('Cannot start: generator not initialized', 'error')
				return
			}

			if (generator.isRunning()) {
				return
			}

			setStatus('generating')
			setError(null)
			setProgress(null)
			setResult(null)
			setLastCheckpoint(null)

			addLog(`Starting generation of ${targetCount.toLocaleString()} primes...`)
			addLog('Using conservative dispatch (32 workgroups × 5 candidates)')

			let lastLoggedCount = 0

			try {
				const data = await generator.generate(
					targetCount,
					prog => {
						setProgress({...prog})

						// Log milestones
						const milestones = [100, 500, 1000, 5000, 10_000, 50_000, 100_000]
						for (const milestone of milestones) {
							if (
								prog.primesFound >= milestone &&
								lastLoggedCount < milestone
							) {
								addLog(
									`Milestone: ${milestone.toLocaleString()} primes found!`,
									'success'
								)
								lastLoggedCount = milestone
							}
						}
					},
					{
						checkpointInterval: 500,
						throttleMs: 16, // ~60fps throttle
						onCheckpoint: (_data, count) => {
							setLastCheckpoint(count)
							addLog(
								`Checkpoint saved: ${count.toLocaleString()} primes`,
								'info'
							)
						}
					}
				)

				setResult(data)
				setStatus('complete')
				addLog(
					`Generation complete! ${targetCount.toLocaleString()} primes generated`,
					'success'
				)
			} catch (err) {
				const message = err instanceof Error ? err.message : 'Unknown error'
				setError(message)
				setStatus('error')
				addLog(`Generation failed: ${message}`, 'error')
			}
		},
		[addLog]
	)

	const stop = useCallback(() => {
		generatorRef.current?.stop()
		addLog('Generation stopped by user', 'warning')
	}, [addLog])

	const download = useCallback(
		(filename?: string) => {
			if (!result) return

			const name = filename || `prime_pool_${Date.now()}.bin`
			// Create a copy of the buffer to ensure it's a proper ArrayBuffer
			const buffer = new ArrayBuffer(result.byteLength)
			new Uint8Array(buffer).set(result)
			const blob = new Blob([buffer], {type: 'application/octet-stream'})
			const url = URL.createObjectURL(blob)

			const a = document.createElement('a')
			a.href = url
			a.download = name
			document.body.appendChild(a)
			a.click()
			document.body.removeChild(a)
			URL.revokeObjectURL(url)

			addLog(`Downloaded: ${name}`, 'success')
		},
		[result, addLog]
	)

	const reset = useCallback(() => {
		generatorRef.current?.stop()
		setStatus(generatorRef.current ? 'ready' : 'idle')
		setError(null)
		setProgress(null)
		setResult(null)
		setLastCheckpoint(null)
		addLog('Reset - ready for new generation')
	}, [addLog])

	return {
		status,
		error,
		gpuInfo,
		progress,
		lastCheckpoint,
		logs,
		result,
		initialize,
		start,
		stop,
		download,
		reset,
		clearLogs
	}
}
