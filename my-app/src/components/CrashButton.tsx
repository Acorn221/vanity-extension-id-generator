import {useState} from 'react'

/**
 * A fun "Crash My PC" button that dispatches an absurdly heavy WebGPU compute shader.
 * This will actually freeze/crash the GPU - use at your own risk!
 */
export function CrashButton() {
	const [status, setStatus] = useState<'idle' | 'armed' | 'loading' | 'rip'>(
		'idle'
	)

	async function crashGPU() {
		if (status === 'idle') {
			setStatus('armed')
			return
		}

		if (status !== 'armed') return

		setStatus('loading')

		if (!navigator.gpu) {
			alert('WebGPU not supported - your PC lives another day')
			setStatus('idle')
			return
		}

		try {
			const adapter = await navigator.gpu.requestAdapter({
				powerPreference: 'high-performance'
			})

			if (!adapter) {
				alert('No GPU adapter found - saved by the hardware gods')
				setStatus('idle')
				return
			}

			const device = await adapter.requestDevice()

			// The crash shader - astronomically long computation
			const shader = device.createShaderModule({
				code: `
					@group(0) @binding(0) var<storage, read_write> data: array<u32>;
					
					@compute @workgroup_size(256)
					fn main(@builtin(global_invocation_id) id: vec3<u32>) {
						var x = id.x;
						// Astronomically long loop - will timeout GPU watchdog
						for (var i = 0u; i < 4294967295u; i++) {
							x = x * 1103515245u + 12345u;
							for (var j = 0u; j < 4294967295u; j++) {
								x = x ^ (x << 13u);
								x = x ^ (x >> 17u);
								x = x ^ (x << 5u);
							}
						}
						data[id.x] = x;
					}
				`
			})

			const buffer = device.createBuffer({
				size: 1024 * 1024 * 4, // 4MB
				usage: GPUBufferUsage.STORAGE
			})

			const pipeline = device.createComputePipeline({
				layout: 'auto',
				compute: {module: shader, entryPoint: 'main'}
			})

			const bindGroup = device.createBindGroup({
				layout: pipeline.getBindGroupLayout(0),
				entries: [{binding: 0, resource: {buffer}}]
			})

			// Dispatch an absurd amount of work
			const encoder = device.createCommandEncoder()
			const pass = encoder.beginComputePass()
			pass.setPipeline(pipeline)
			pass.setBindGroup(0, bindGroup)
			pass.dispatchWorkgroups(65535, 65535, 1) // Maximum dispatch
			pass.end()

			device.queue.submit([encoder.finish()])

			setStatus('rip')
			console.log('RIP your GPU')
		} catch (error) {
			console.error('Failed to crash:', error)
			alert(`Crash failed: ${error}`)
			setStatus('idle')
		}
	}

	function getButtonText() {
		switch (status) {
			case 'idle':
				return 'Crash My PC'
			case 'armed':
				return 'Are you sure? Click again!'
			case 'loading':
				return 'Initializing doom...'
			case 'rip':
				return 'RIP'
		}
	}

	function getButtonStyle() {
		const base =
			'px-6 py-3 rounded-xl font-bold text-white transition-all duration-200 cursor-pointer shadow-lg'

		switch (status) {
			case 'idle':
				return `${base} bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 hover:scale-105 shadow-red-500/40`
			case 'armed':
				return `${base} bg-gradient-to-r from-orange-500 to-red-500 hover:scale-105 animate-pulse shadow-orange-500/50`
			case 'loading':
				return `${base} bg-gradient-to-r from-yellow-500 to-orange-500 cursor-wait`
			case 'rip':
				return `${base} bg-gradient-to-r from-gray-700 to-gray-800 cursor-not-allowed`
		}
	}

	return (
		<div className='fixed bottom-4 right-4 z-50 flex flex-col items-end gap-2'>
			<button
				className={getButtonStyle()}
				disabled={status === 'loading' || status === 'rip'}
				onClick={crashGPU}
				type='button'
			>
				{status === 'idle' || status === 'armed' ? '🔥 ' : ''}
				{getButtonText()}
				{status === 'idle' || status === 'armed' ? ' 🔥' : ''}
				{status === 'rip' ? ' 💀' : ''}
			</button>
			<p className='text-xs text-red-400/70'>
				{status === 'armed'
					? '⚠️ This WILL freeze your computer'
					: status === 'rip'
						? 'Check if you can still read this...'
						: '⚠️ Requires WebGPU'}
			</p>
		</div>
	)
}





