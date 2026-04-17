import Link from 'next/link'

export default function Home() {
  return (
    <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      {/* Navigation */}
      <nav className="container-w flex items-center justify-between border-b border-slate-700 py-4">
        <div className="flex items-center gap-2">
          <div className="rounded-lg bg-blue-600 px-3 py-2 font-bold">MS</div>
          <h1 className="text-xl font-bold">MindScope</h1>
        </div>
        <div className="flex gap-6">
          <a href="#about" className="text-slate-300 hover:text-white transition-colors">
            About
          </a>
          <a href="#demo" className="text-slate-300 hover:text-white transition-colors">
            Demo
          </a>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="container-w space-y-8 py-20 text-center">
        <h2 className="text-5xl font-bold leading-tight">
          See What Your Brain Sees
        </h2>
        <p className="mx-auto max-w-2xl text-xl text-slate-300">
          Upload a video. Our AI predicts how your visual cortex, auditory cortex,
          and language areas would respond. Ask Gemma to explain what&apos;s happening.
        </p>

        <div className="flex flex-col gap-4 sm:flex-row justify-center sm:gap-6">
          <Link href="/demo" className="btn btn-primary text-lg px-8 py-4">
            Try the Demo →
          </Link>
          <a href="#about" className="btn btn-secondary text-lg px-8 py-4">
            Learn More
          </a>
        </div>
      </section>

      {/* Feature Grid */}
      <section id="about" className="container-w space-y-8 py-16">
        <h3 className="text-3xl font-bold text-center">Powered By</h3>

        <div className="grid gap-6 sm:grid-cols-3">
          {/* TRIBE v2 */}
          <div className="card bg-slate-800 border-slate-700">
            <h4 className="text-lg font-semibold text-blue-400 mb-2">TRIBE v2</h4>
            <p className="text-slate-300">
              Meta&apos;s brain-response foundation model. Predicts activity on 20,484 cortical vertices at 2 Hz.
            </p>
          </div>

          {/* Gemma 4 E4B */}
          <div className="card bg-slate-800 border-slate-700">
            <h4 className="text-lg font-semibold text-blue-400 mb-2">Gemma 4 E4B</h4>
            <p className="text-slate-300">
              Google&apos;s multimodal LLM. Runs on your laptop. Explains brain predictions in plain language.
            </p>
          </div>

          {/* Offline First */}
          <div className="card bg-slate-800 border-slate-700">
            <h4 className="text-lg font-semibold text-blue-400 mb-2">Offline Ready</h4>
            <p className="text-slate-300">
              HIPAA-compliant. No data leaves your hospital. Clinics with no internet can still use it.
            </p>
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="container-w space-y-8 py-16">
        <h3 className="text-3xl font-bold text-center">For Clinicians, Researchers & Students</h3>

        <div className="grid gap-8 sm:grid-cols-2">
          {/* Clinicians */}
          <div className="space-y-4">
            <h4 className="text-xl font-semibold">Clinicians</h4>
            <p className="text-slate-300">
              Screen content for seizure triggers, photosensitivity, and sensory overload.
              Generate accessibility reports in seconds.
            </p>
          </div>

          {/* Researchers */}
          <div className="space-y-4">
            <h4 className="text-xl font-semibold">Researchers</h4>
            <p className="text-slate-300">
              Understand how visual, auditory, and language features drive cortical responses.
              Export predictions for downstream analysis.
            </p>
          </div>

          {/* Students */}
          <div className="space-y-4">
            <h4 className="text-xl font-semibold">Educators</h4>
            <p className="text-slate-300">
              Intro neuroscience labs: watch real-time cortex activation as students learn.
              Measure video engagement for remote learning.
            </p>
          </div>

          {/* Healthcare Systems */}
          <div className="space-y-4">
            <h4 className="text-xl font-semibold">Health Systems</h4>
            <p className="text-slate-300">
              Deploy inside your hospital VPC. No PHI leaves your network.
              HIPAA audit logs included.
            </p>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section id="demo" className="container-w space-y-8 py-16 text-center">
        <h3 className="text-3xl font-bold">Ready to explore?</h3>
        <p className="text-slate-300">
          Try the demo with a pre-loaded cat video (20 seconds), then upload your own clips.
        </p>
        <Link href="/demo" className="inline-block btn btn-primary text-lg px-8 py-4">
          Launch Demo →
        </Link>
      </section>

      {/* Footer */}
      <footer className="container-w border-t border-slate-700 py-8 text-center text-slate-400">
        <p>
          MindScope × Gemma 4 Good Hackathon. Built with{' '}
          <a href="https://github.com/facebookresearch/tribev2" className="text-blue-400 hover:text-blue-300">
            TRIBE v2
          </a>
          {' '}&{' '}
          <a href="https://google.com/gemma" className="text-blue-400 hover:text-blue-300">
            Gemma
          </a>
          .
        </p>
      </footer>
    </div>
  )
}
