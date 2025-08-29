'use client'

import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { useQuantum } from '@/providers/QuantumProvider'

interface SuperpositionState {
  id: string
  symbol: string
  position: 'long' | 'short' | 'neutral'
  probability: number
  expectedReturn: number
  risk: number
  collapsed: boolean
}

export default function PortfolioSuperposition() {
  const { quantumState, collapseWaveFunction, measurePortfolio } = useQuantum()
  const [states, setStates] = useState<SuperpositionState[]>([])
  const [isCollapsing, setIsCollapsing] = useState(false)

  useEffect(() => {
    // Generate superposition states
    const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META']
    const positions: Array<'long' | 'short' | 'neutral'> = ['long', 'short', 'neutral']
    
    const newStates = Array(8).fill(null).map((_, i) => ({
      id: `state-${i}`,
      symbol: symbols[Math.floor(Math.random() * symbols.length)],
      position: positions[Math.floor(Math.random() * positions.length)],
      probability: Math.random() * 0.3 + 0.1,
      expectedReturn: (Math.random() - 0.5) * 20,
      risk: Math.random() * 10 + 2,
      collapsed: false,
    }))
    
    // Normalize probabilities
    const totalProb = newStates.reduce((sum, s) => sum + s.probability, 0)
    newStates.forEach(s => s.probability = s.probability / totalProb)
    
    setStates(newStates)
  }, [])

  const handleCollapse = () => {
    setIsCollapsing(true)
    collapseWaveFunction('user')
    
    setTimeout(() => {
      const measured = measurePortfolio()
      setStates(prev => prev.map(s => ({
        ...s,
        collapsed: Math.random() < s.probability,
      })))
      setIsCollapsing(false)
    }, 1000)
  }

  const positionColors = {
    long: 'text-green-400 border-green-400',
    short: 'text-red-400 border-red-400',
    neutral: 'text-gray-400 border-gray-400',
  }

  return (
    <div className="quantum-glass rounded-xl p-4 h-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold quantum-text">Portfolio Superposition</h3>
        <button
          onClick={handleCollapse}
          disabled={isCollapsing}
          className="px-3 py-1 rounded-lg bg-quantum-execution/20 hover:bg-quantum-execution/30 
            text-quantum-execution text-sm font-semibold transition-all duration-200
            disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isCollapsing ? 'Collapsing...' : 'Observe'}
        </button>
      </div>

      <div className="space-y-2 max-h-[220px] overflow-y-auto">
        {states.map((state, i) => (
          <motion.div
            key={state.id}
            initial={{ x: -20, opacity: 0 }}
            animate={{ 
              x: 0, 
              opacity: state.collapsed ? 1 : 0.3 + state.probability,
              scale: state.collapsed ? 1.05 : 1,
            }}
            transition={{ delay: i * 0.05 }}
            className={`relative p-2 rounded-lg border ${
              state.collapsed ? 'border-quantum-execution bg-quantum-execution/10' : 'border-gray-700'
            } transition-all duration-300`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-sm font-mono font-semibold">{state.symbol}</span>
                <span className={`text-xs px-2 py-0.5 rounded border ${positionColors[state.position]}`}>
                  {state.position.toUpperCase()}
                </span>
              </div>
              
              <div className="flex items-center gap-4 text-xs">
                <div>
                  <span className="text-gray-500">P: </span>
                  <span className="text-quantum-cyan font-mono">
                    {(state.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">E[R]: </span>
                  <span className={state.expectedReturn > 0 ? 'text-green-400' : 'text-red-400'}>
                    {state.expectedReturn > 0 ? '+' : ''}{state.expectedReturn.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Probability Wave Visualization */}
            <div className="absolute inset-0 pointer-events-none">
              <svg className="w-full h-full">
                <path
                  d={`M 0,${20} Q ${50 * state.probability},${10} 100,${20}`}
                  stroke="rgba(0, 255, 255, 0.3)"
                  strokeWidth="1"
                  fill="none"
                  className={isCollapsing ? 'animate-collapse' : 'animate-wave-function'}
                />
              </svg>
            </div>

            {state.collapsed && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="absolute -right-2 -top-2 w-4 h-4 bg-quantum-execution rounded-full"
              />
            )}
          </motion.div>
        ))}
      </div>

      {/* Quantum Statistics */}
      <div className="mt-4 pt-3 border-t border-gray-700 flex justify-around text-xs">
        <div>
          <span className="text-gray-500">States: </span>
          <span className="text-quantum-cyan font-mono">{states.length}</span>
        </div>
        <div>
          <span className="text-gray-500">Entropy: </span>
          <span className="text-quantum-energy-high font-mono">
            {(-states.reduce((sum, s) => 
              sum + (s.probability * Math.log2(s.probability || 1)), 0
            )).toFixed(2)}
          </span>
        </div>
        <div>
          <span className="text-gray-500">Coherence: </span>
          <span className="text-quantum-execution font-mono">
            {(100 - states.filter(s => s.collapsed).length * 12.5).toFixed(0)}%
          </span>
        </div>
      </div>
    </div>
  )
}