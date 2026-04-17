'use client'

import { useState } from 'react'
import Link from 'next/link'

export default function DemoPage() {
  const [jobId, setJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)

  const handleDemoClick = async () => {
    setIsProcessing(true)
    setProgress(0)

    // TODO: Integrate with FastAPI backend
    // For now, simulate progress
    const interval = setInterval(() => {
      setProgress(p => {
        if (p >= 90) {
          clearInterval(interval)
          return p
        }
        return p + Math.random() * 20
      })
    }, 500)

    // Simulate job submission
    setTimeout(() => {
      setJobId('demo-cat-video-1')
      setProgress(100)
      setIsProcessing(false)
      clearInterval(interval)
    }, 3000)
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <nav className="container-w flex items-center justify-between border-b border-slate-200 bg-white py-4">
        <Link href="/" className="flex items-center gap-2 font-bold">
          <div className="rounded-lg bg-blue-600 px-3 py-2 text-white">MS</div>
          <span>MindScope</span>
        </Link>
        <div className="flex gap-4">
          <Link href="/" className="text-slate-600 hover:text-slate-900">
            Home
          </Link>
        </div>
      </nav>

      {/* Main */}
      <main className="container-w py-12">
        <h1 className="text-4xl font-bold mb-2">Brain-Response Prediction Demo</h1>
        <p className="text-slate-600 mb-12 text-lg">
          Upload a video (or try the preset cat demo), and see what the average human brain predicts.
        </p>

        <div className="grid gap-12 lg:grid-cols-2">
          {/* Upload Panel */}
          <div className="space-y-6">
            <div className="card">
              <h2 className="text-2xl font-bold mb-4">Try the Preset Demo</h2>
              <p className="text-slate-600 mb-6">
                20-second clip of a cat with purring audio and English narration.
              </p>
              <button
                onClick={handleDemoClick}
                disabled={isProcessing}
                className="btn btn-primary w-full text-lg py-3"
              >
                {isProcessing ? 'Processing...' : 'Run Demo'}
              </button>

              {isProcessing && (
                <div className="mt-6 space-y-2">
                  <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-600 transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <p className="text-sm text-slate-600">{Math.round(progress)}% complete</p>
                </div>
              )}
            </div>

            <div className="card">
              <h2 className="text-2xl font-bold mb-4">Or Upload Your Own</h2>
              <p className="text-slate-600 mb-6">
                Video file up to 60 seconds, MP4 format.
              </p>
              <div className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center">
                <p className="text-slate-600 mb-4">Drag & drop your video here, or click to browse</p>
                <input
                  type="file"
                  accept="video/mp4"
                  className="hidden"
                  id="video-upload"
                />
                <label
                  htmlFor="video-upload"
                  className="btn btn-secondary cursor-pointer"
                >
                  Choose File
                </label>
              </div>
              <p className="text-xs text-slate-500 mt-4">
                Note: Feature upload coming soon. For now, try the demo above.
              </p>
            </div>
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {jobId ? (
              <>
                <div className="card bg-blue-50 border-blue-200">
                  <h2 className="text-2xl font-bold mb-2 text-blue-900">Results Ready!</h2>
                  <p className="text-blue-700 mb-4">Job ID: {jobId}</p>
                  <button className="btn btn-primary">View Full Results →</button>
                </div>

                <div className="card">
                  <h3 className="text-lg font-semibold mb-4">Predicted Brain Activity</h3>
                  <div className="space-y-4">
                    <div>
                      <p className="text-sm text-slate-600 mb-2">Visual Cortex (V1/V2)</p>
                      <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                        <div className="h-full w-3/4 bg-blue-500" />
                      </div>
                    </div>
                    <div>
                      <p className="text-sm text-slate-600 mb-2">Auditory Cortex (A1)</p>
                      <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                        <div className="h-full w-2/3 bg-green-500" />
                      </div>
                    </div>
                    <div>
                      <p className="text-sm text-slate-600 mb-2">Language Areas (Broca, Wernicke)</p>
                      <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                        <div className="h-full w-1/2 bg-purple-500" />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="card">
                  <h3 className="text-lg font-semibold mb-4">Gemma Narration</h3>
                  <p className="text-slate-700 leading-relaxed">
                    &quot;At 0–2 seconds, the cat enters the frame. Your visual cortex spikes as the striped fur pattern activates orientation-selective neurons. The purring sound at 50 Hz engages your auditory cortex. The narration &apos;Look at the cat&apos; lights up Broca's area and Wernicke's area as you parse the semantics. The tail movement at 5 seconds causes sustained activation in MT/MST (motion areas).&quot;
                  </p>
                </div>

                <div className="card">
                  <h3 className="text-lg font-semibold mb-4">Ask Gemma a Question</h3>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      placeholder="Why did my visual cortex spike at 3 seconds?"
                      className="flex-1 px-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <button className="btn btn-primary">Ask</button>
                  </div>
                </div>
              </>
            ) : (
              <div className="card h-96 flex items-center justify-center">
                <p className="text-center text-slate-500">
                  Run the demo or upload a video to see results here.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
