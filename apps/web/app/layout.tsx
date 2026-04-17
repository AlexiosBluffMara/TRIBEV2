import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'MindScope',
  description: 'Brain-response prediction × Gemma 4 E4B for clinicians, researchers, and students',
  icons: {
    icon: '/favicon.ico',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="bg-slate-50 text-slate-900">
        <div className="min-h-screen">
          {children}
        </div>
      </body>
    </html>
  )
}
