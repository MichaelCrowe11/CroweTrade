import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { QuantumProvider } from '@/providers/QuantumProvider'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })

export const metadata: Metadata = {
  title: 'CroweTrade Quantum | Where Mathematics Meets Market Reality',
  description: 'Revolutionary quantum-inspired trading platform with advanced visualization and AI-driven insights',
  keywords: 'quantum trading, algorithmic trading, market visualization, AI trading',
  authors: [{ name: 'CroweTrade' }],
  viewport: 'width=device-width, initial-scale=1',
  themeColor: '#001F3F',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen bg-quantum-void overflow-x-hidden">
        <QuantumProvider>
          <div className="quantum-field fixed inset-0 pointer-events-none" />
          <div className="wave-pattern fixed inset-0 opacity-20 pointer-events-none" />
          {children}
        </QuantumProvider>
      </body>
    </html>
  )
}