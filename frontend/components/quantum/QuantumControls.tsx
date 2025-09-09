'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'

interface QuantumParameter {
  name: string
  value: number
  min: number
  max: number
  unit: string
  entangled: string[]
}

export default function QuantumControls() {
  const [parameters, setParameters] = useState<QuantumParameter[]>([
    { name: 'Risk Amplitude', value: 50, min: 0, max: 100, unit: '%', entangled: ['Position Size', 'Stop Loss'] },
    { name: 'Position Size', value: 30, min: 0, max: 100, unit: '%', entangled: ['Risk Amplitude'] },
    { name: 'Time Horizon', value: 24, min: 1, max: 168, unit: 'h', entangled: ['Volatility Window'] },
    { name: 'Volatility Window', value: 20, min: 5, max: 60, unit: 'd', entangled: ['Time Horizon'] },
    { name: 'Coherence Threshold', value: 75, min: 50, max: 100, unit: '%', entangled: [] },
  ])

  const handleParameterChange = (index: number, newValue: number) => {
    setParameters(prev => {
      const updated = [...prev]
      updated[index].value = newValue
      
      // Update entangled parameters
      updated[index].entangled.forEach(entangledName => {
        const entangledIndex = updated.findIndex(p => p.name === entangledName)
        if (entangledIndex !== -1) {
          // Quantum entanglement effect
          const influence = (newValue - prev[index].value) * 0.3
          updated[entangledIndex].value = Math.max(
            updated[entangledIndex].min,
            Math.min(
              updated[entangledIndex].max,
              updated[entangledIndex].value + influence
            )
          )
        }
      })
      
      return updated
    })
  }

  return (
    <div className="quantum-glass rounded-xl p-4">
      <h3 className="text-lg font-semibold quantum-text mb-4">Quantum Controls</h3>
      
      <div className="space-y-4">
        {parameters.map((param, index) => (
          <div key={param.name} className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-300">{param.name}</span>
              <span className="text-sm font-mono text-quantum-cyan">
                {param.value.toFixed(0)}{param.unit}
              </span>
            </div>
            
            <div className="relative">
              {/* Quantum Slider Track */}
              <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-quantum-energy-high via-quantum-cyan to-quantum-execution"
                  style={{ width: `${(param.value - param.min) / (param.max - param.min) * 100}%` }}
                  animate={{ width: `${(param.value - param.min) / (param.max - param.min) * 100}%` }}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              </div>
              
              {/* Quantum Slider Handle */}
              <input
                type="range"
                min={param.min}
                max={param.max}
                value={param.value}
                onChange={(e) => handleParameterChange(index, Number(e.target.value))}
                className="absolute inset-0 w-full opacity-0 cursor-pointer"
              />
              
              <motion.div
                className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-quantum-cyan rounded-full
                  shadow-lg shadow-quantum-cyan/50 pointer-events-none"
                style={{ left: `${(param.value - param.min) / (param.max - param.min) * 100}%` }}
                animate={{ left: `${(param.value - param.min) / (param.max - param.min) * 100}%` }}
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
              >
                <div className="absolute inset-0 bg-quantum-cyan rounded-full animate-ping" />
              </motion.div>
            </div>
            
            {/* Entanglement Indicators */}
            {param.entangled.length > 0 && (
              <div className="flex items-center gap-1 mt-1">
                <span className="text-xs text-gray-500">Entangled:</span>
                {param.entangled.map(name => (
                  <span key={name} className="text-xs px-1.5 py-0.5 rounded bg-quantum-cyan/10 
                    text-quantum-cyan border border-quantum-cyan/30">
                    {name}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
      
      {/* Quantum State Indicator */}
      <div className="mt-6 p-3 rounded-lg bg-gradient-to-r from-quantum-energy-high/10 
        via-quantum-cyan/10 to-quantum-execution/10 border border-quantum-cyan/30">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-400">Quantum Coherence</span>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-quantum-execution rounded-full animate-pulse" />
            <span className="text-sm font-mono text-quantum-execution">STABLE</span>
          </div>
        </div>
      </div>
    </div>
  )
}