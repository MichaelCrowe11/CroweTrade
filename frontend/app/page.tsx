'use client'

import dynamic from 'next/dynamic'
import { Suspense } from 'react'
import AgentConsciousnessGrid from '@/components/quantum/AgentConsciousnessGrid'
import PortfolioSuperposition from '@/components/quantum/PortfolioSuperposition'
import QuantumControls from '@/components/quantum/QuantumControls'
import MarketHeatmap from '@/components/quantum/MarketHeatmap'
import RiskQuantumStates from '@/components/quantum/RiskQuantumStates'

const QuantumField = dynamic(() => import('@/components/quantum/QuantumField'), {
  ssr: false,
  loading: () => <div className="w-full h-full bg-quantum-void animate-pulse" />
})

export default function QuantumTradingCockpit() {
  return (
    <main className="min-h-screen p-4 space-y-4">
      {/* Header */}
      <header className="quantum-glass rounded-xl p-6 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 relative">
            <div className="absolute inset-0 bg-gradient-to-br from-quantum-cyan to-quantum-energy-high rounded-lg animate-quantum-pulse" />
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-2xl font-bold">🌌</span>
            </div>
          </div>
          <div>
            <h1 className="text-3xl font-bold quantum-text">CroweTrade Quantum</h1>
            <p className="text-sm text-gray-400">Where Mathematics Meets Market Reality</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="text-right">
            <div className="text-xs text-gray-400">System Coherence</div>
            <div className="text-2xl font-mono text-quantum-execution">98.7%</div>
          </div>
          <div className="text-right">
            <div className="text-xs text-gray-400">Quantum State</div>
            <div className="text-lg font-mono text-quantum-cyan">SUPERPOSITION</div>
          </div>
        </div>
      </header>

      {/* Main Grid Layout */}
      <div className="grid grid-cols-12 gap-4">
        {/* Left Column - Quantum Field & Portfolio */}
        <div className="col-span-4 space-y-4">
          <div className="h-[400px] quantum-glass rounded-xl p-4">
            <h3 className="text-lg font-semibold mb-2 quantum-text">Market Quantum Field</h3>
            <div className="h-[340px]">
              <Suspense fallback={<div>Loading quantum field...</div>}>
                <QuantumField />
              </Suspense>
            </div>
          </div>
          
          <div className="h-[300px]">
            <PortfolioSuperposition />
          </div>
        </div>

        {/* Center Column - Agent Grid */}
        <div className="col-span-5">
          <AgentConsciousnessGrid />
        </div>

        {/* Right Column - Controls & Risk */}
        <div className="col-span-3 space-y-4">
          <QuantumControls />
          <RiskQuantumStates />
        </div>
      </div>

      {/* Bottom Section - Market Heatmap */}
      <div className="quantum-glass rounded-xl p-4 h-[200px]">
        <MarketHeatmap />
      </div>

      {/* Floating Notifications */}
      <div className="fixed bottom-4 right-4 w-80 space-y-2">
        <motion.div
          initial={{ x: 400, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="quantum-glass rounded-lg p-3 border-l-4 border-quantum-execution"
        >
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold">Wave Function Collapsed</span>
            <span className="text-xs text-gray-400">Just now</span>
          </div>
          <p className="text-xs text-gray-300 mt-1">
            Portfolio state measured: Long AAPL, Short TSLA
          </p>
        </motion.div>
      </div>
    </main>
  )
}

import { motion } from 'framer-motion'