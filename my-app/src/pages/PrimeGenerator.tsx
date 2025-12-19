import {useEffect, useRef, useState} from 'react'
import {Head} from '@/components/Head'
import {PRIME_BYTES} from '@/gpu'
import {usePrimeGenerator} from '@/hooks/usePrimeGenerator'

const PRESETS = [
	{label: '1K', value: 1000},
	{label: '5K', value: 5000},
	{label: '10K', value: 10_000},
	{label: '50K', value: 50_000},
	{label: '100K', value: 100_000}
]

function formatNumber(n: number): string {
	if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
	if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
	return n.toString()
}

function formatTime(seconds: number): string {
	if (seconds === Number.POSITIVE_INFINITY) return '—'
	if (seconds < 60) return `${Math.round(seconds)}s`
	if (seconds < 3600) {
		const m = Math.floor(seconds / 60)
		const s = Math.round(seconds % 60)
		return `${m}m ${s}s`
	}
	const h = Math.floor(seconds / 3600)
	const m = Math.floor((seconds % 3600) / 60)
	return `${h}h ${m}m`
}

function formatBytes(bytes: number): string {
	if (bytes >= 1024 * 1024 * 1024)
		return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`
	if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
	if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`
	return `${bytes} B`
}

function formatTimestamp(date: Date): string {
	return date.toLocaleTimeString('en-US', {
		hour12: false,
		hour: '2-digit',
		minute: '2-digit',
		second: '2-digit'
	})
}

export function PrimeGenerator() {
	const {
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
	} = usePrimeGenerator()

	const [targetCount, setTargetCount] = useState(10_000)
	const [customInput, setCustomInput] = useState('')
	const logContainerRef = useRef<HTMLDivElement>(null)

	// Auto-initialize on mount
	useEffect(() => {
		if (status === 'idle') {
			initialize()
		}
	}, [status, initialize])

	// Auto-scroll logs
	useEffect(() => {
		if (logContainerRef.current) {
			logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
		}
	}, [])

	const handleStart = () => {
		const count = customInput ? Number.parseInt(customInput, 10) : targetCount
		if (count > 0) {
			start(count)
		}
	}

	const handlePreset = (value: number) => {
		setTargetCount(value)
		setCustomInput('')
	}

	const progressPercent = progress
		? (progress.primesFound / progress.targetCount) * 100
		: 0

	const estimatedSize =
		(customInput ? Number.parseInt(customInput, 10) : targetCount) *
			PRIME_BYTES +
		16

	const getLogColor = (type: string) => {
		switch (type) {
			case 'success':
				return 'text-emerald-400'
			case 'warning':
				return 'text-yellow-400'
			case 'error':
				return 'text-red-400'
			default:
				return 'text-slate-400'
		}
	}

	return (
		<>
			<Head title='Prime Generator | WebGPU' />
			<div className='min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100'>
				{/* Ambient glow effect */}
				<div className='pointer-events-none fixed inset-0 overflow-hidden'>
					<div className='-top-1/2 -translate-x-1/2 absolute left-1/2 h-[800px] w-[800px] rounded-full bg-emerald-500/10 blur-[128px]' />
					<div className='-bottom-1/2 absolute right-0 h-[600px] w-[600px] rounded-full bg-cyan-500/10 blur-[128px]' />
				</div>

				<div className='relative mx-auto max-w-2xl px-6 py-16'>
					{/* Header */}
					<header className='mb-12 text-center'>
						<h1 className='mb-3 font-bold text-4xl tracking-tight'>
							<span className='bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent'>
								Prime Pool Generator
							</span>
						</h1>
						<p className='text-slate-400'>
							WebGPU-accelerated 1024-bit prime generation
						</p>
					</header>

					{/* GPU Info Card */}
					<div className='mb-6 rounded-2xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm'>
						<div className='mb-4 flex items-center gap-3'>
							<div
								className={`h-3 w-3 rounded-full ${
									status === 'error'
										? 'bg-red-500'
										: gpuInfo?.available
											? 'bg-emerald-500 shadow-emerald-500/50 shadow-lg'
											: 'animate-pulse bg-yellow-500'
								}`}
							/>
							<span className='font-medium text-slate-300 text-sm'>
								{status === 'initializing' && 'Initializing WebGPU...'}
								{status === 'error' && 'WebGPU Error'}
								{status === 'generating' && 'Generating...'}
								{status === 'complete' && 'Complete!'}
								{status === 'ready' && gpuInfo?.deviceName}
								{status === 'idle' && 'Checking GPU...'}
							</span>
						</div>

						{error && (
							<div className='rounded-lg border border-red-500/20 bg-red-500/10 p-4 text-red-400 text-sm'>
								{error}
							</div>
						)}

						{gpuInfo?.available && status !== 'generating' && (
							<div className='grid grid-cols-2 gap-4 text-sm'>
								<div>
									<span className='text-slate-500'>Max Workgroup</span>
									<p className='font-mono text-slate-300'>
										{gpuInfo.maxWorkgroupSize}
									</p>
								</div>
								<div>
									<span className='text-slate-500'>Max Buffer</span>
									<p className='font-mono text-slate-300'>
										{formatBytes(gpuInfo.maxBufferSize)}
									</p>
								</div>
							</div>
						)}
					</div>

					{/* Controls */}
					{(status === 'ready' || status === 'complete') && (
						<div className='mb-6 rounded-2xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm'>
							<h2 className='mb-4 font-semibold text-lg text-slate-200'>
								Target Count
							</h2>

							{/* Presets */}
							<div className='mb-4 flex flex-wrap gap-2'>
								{PRESETS.map(preset => (
									<button
										className={`rounded-lg px-4 py-2 font-medium text-sm transition-all ${
											targetCount === preset.value && !customInput
												? 'bg-emerald-500 text-white shadow-emerald-500/25 shadow-lg'
												: 'bg-slate-800 text-slate-300 hover:bg-slate-700'
										}`}
										key={preset.value}
										onClick={() => handlePreset(preset.value)}
										type='button'
									>
										{preset.label}
									</button>
								))}
							</div>

							{/* Custom input */}
							<div className='mb-6'>
								<input
									className='w-full rounded-lg border border-slate-700 bg-slate-800/50 px-4 py-3 text-slate-200 placeholder-slate-500 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500'
									onChange={e => setCustomInput(e.target.value)}
									placeholder='Custom count...'
									type='number'
									value={customInput}
								/>
								<p className='mt-2 text-slate-500 text-sm'>
									Estimated file size:{' '}
									<span className='text-slate-400'>
										{formatBytes(estimatedSize)}
									</span>
								</p>
							</div>

							{/* Start button */}
							<button
								className='w-full rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 py-4 font-semibold text-lg text-white shadow-emerald-500/25 shadow-lg transition-all hover:shadow-emerald-500/30 hover:shadow-xl active:scale-[0.98]'
								onClick={handleStart}
								type='button'
							>
								Generate Primes
							</button>
						</div>
					)}

					{/* Progress */}
					{status === 'generating' && progress && (
						<div className='mb-6 rounded-2xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm'>
							<div className='mb-4 flex items-center justify-between'>
								<div className='flex items-center gap-3'>
									<div className='h-3 w-3 animate-pulse rounded-full bg-emerald-500' />
									<span className='font-semibold text-lg text-slate-200'>
										Generating...
									</span>
								</div>
								<button
									className='rounded-lg bg-red-500/20 px-4 py-2 font-medium text-red-400 text-sm transition-colors hover:bg-red-500/30'
									onClick={stop}
									type='button'
								>
									Stop
								</button>
							</div>

							{/* Progress bar */}
							<div className='relative mb-4 h-4 overflow-hidden rounded-full bg-slate-800'>
								<div
									className='absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-emerald-500 to-cyan-500 transition-all duration-300'
									style={{width: `${Math.min(progressPercent, 100)}%`}}
								/>
							</div>

							{/* Stats grid */}
							<div className='mb-4 grid grid-cols-2 gap-4 text-center sm:grid-cols-4'>
								<div className='rounded-lg bg-slate-800/50 p-3'>
									<p className='font-bold text-2xl text-emerald-400'>
										{formatNumber(progress.primesFound)}
									</p>
									<p className='text-slate-500 text-xs'>Found</p>
								</div>
								<div className='rounded-lg bg-slate-800/50 p-3'>
									<p className='font-bold text-2xl text-slate-200'>
										{progressPercent.toFixed(1)}%
									</p>
									<p className='text-slate-500 text-xs'>Progress</p>
								</div>
								<div className='rounded-lg bg-slate-800/50 p-3'>
									<p className='font-bold text-2xl text-cyan-400'>
										{formatNumber(Math.round(progress.rate))}
									</p>
									<p className='text-slate-500 text-xs'>Primes/s</p>
								</div>
								<div className='rounded-lg bg-slate-800/50 p-3'>
									<p className='font-bold text-2xl text-slate-200'>
										{formatTime(progress.eta)}
									</p>
									<p className='text-slate-500 text-xs'>ETA</p>
								</div>
							</div>

							{/* Checkpoint indicator */}
							{lastCheckpoint && (
								<div className='flex items-center gap-2 rounded-lg bg-blue-500/10 p-3 text-blue-400 text-sm'>
									<svg
										className='h-4 w-4'
										fill='none'
										stroke='currentColor'
										viewBox='0 0 24 24'
									>
										<path
											d='M5 13l4 4L19 7'
											strokeLinecap='round'
											strokeLinejoin='round'
											strokeWidth={2}
										/>
									</svg>
									<span>
										Last checkpoint: {lastCheckpoint.toLocaleString()} primes
										saved
									</span>
								</div>
							)}
						</div>
					)}

					{/* Complete */}
					{status === 'complete' && result && (
						<div className='mb-6 rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-6 backdrop-blur-sm'>
							<div className='mb-4 flex items-center gap-3'>
								<div className='flex h-10 w-10 items-center justify-center rounded-full bg-emerald-500/20'>
									<svg
										aria-label='Success checkmark'
										className='h-5 w-5 text-emerald-400'
										fill='none'
										role='img'
										stroke='currentColor'
										viewBox='0 0 24 24'
									>
										<path
											d='M5 13l4 4L19 7'
											strokeLinecap='round'
											strokeLinejoin='round'
											strokeWidth={2}
										/>
									</svg>
								</div>
								<div>
									<h3 className='font-semibold text-emerald-400'>
										Generation Complete!
									</h3>
									<p className='text-slate-400 text-sm'>
										{progress
											? `${formatNumber(progress.primesFound)} primes in ${formatTime(progress.elapsed)}`
											: 'Ready to download'}
									</p>
								</div>
							</div>

							<div className='flex gap-3'>
								<button
									className='flex-1 rounded-xl bg-emerald-500 py-3 font-semibold text-white transition-all hover:bg-emerald-400 active:scale-[0.98]'
									onClick={() => download()}
									type='button'
								>
									Download .bin ({formatBytes(result.byteLength)})
								</button>
								<button
									className='rounded-xl bg-slate-700 px-6 py-3 font-semibold text-slate-200 transition-all hover:bg-slate-600 active:scale-[0.98]'
									onClick={reset}
									type='button'
								>
									New
								</button>
							</div>
						</div>
					)}

					{/* Activity Log */}
					<div className='rounded-2xl border border-slate-800 bg-slate-900/50 backdrop-blur-sm'>
						<div className='flex items-center justify-between border-slate-800 border-b px-4 py-3'>
							<h3 className='font-medium text-slate-300 text-sm'>
								Activity Log
							</h3>
							<button
								className='text-slate-500 text-xs hover:text-slate-400'
								onClick={clearLogs}
								type='button'
							>
								Clear
							</button>
						</div>
						<div
							className='h-48 overflow-y-auto p-4 font-mono text-xs'
							ref={logContainerRef}
						>
							{logs.length === 0 ? (
								<p className='text-slate-600'>No activity yet...</p>
							) : (
								<div className='space-y-1'>
									{logs.map(log => (
										<div className='flex gap-2' key={log.id}>
											<span className='text-slate-600'>
												[{formatTimestamp(log.timestamp)}]
											</span>
											<span className={getLogColor(log.type)}>
												{log.message}
											</span>
										</div>
									))}
								</div>
							)}
						</div>
					</div>

					{/* Footer info */}
					<footer className='mt-8 text-center text-slate-500 text-sm'>
						<p>
							Output format: PRMP v1 • 1024-bit primes • 64 Miller-Rabin rounds
						</p>
						<p className='mt-1'>Compatible with vanity-ext-id C++ tools</p>
					</footer>
				</div>
			</div>
		</>
	)
}
