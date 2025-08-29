'use client'

import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

interface RiskLevel {
  name: string
  energy: number
  probability: number
  color: string
  threshold: number
}

export default function RiskQuantumStates() {
  const [riskLevels] = useState<RiskLevel[]>([
    { name: 'Ground State', energy: 0, probability: 0.4, color: '#39FF14', threshold: 0 },
    { name: 'First Excited', energy: 25, probability: 0.3, color: '#00FFFF', threshold: 25 },
    { name: 'Second Excited', energy: 50, probability: 0.2, color: '#FFFF00', threshold: 50 },
    { name: 'Third Excited', energy: 75, probability: 0.08, color: '#FF8800', threshold: 75 },
    { name: 'Critical State', energy: 100, probability: 0.02, color: '#FF0000', threshold: 90 },
  ])
  
  const [currentEnergy, setCurrentEnergy] = useState(15)
  const [transitions, setTransitions] = useState<Array<{from: number, to: number, time: number}>>([])

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentEnergy(prev => {
        // Random walk with mean reversion
        const randomChange = (Math.random() - 0.5) * 10
        const meanReversion = (30 - prev) * 0.1
        const newEnergy = Math.max(0, Math.min(100, prev + randomChange + meanReversion))
        
        // Record transitions
        const prevLevel = riskLevels.findIndex(l => prev < l.threshold + 12.5)
        const newLevel = riskLevels.findIndex(l => newEnergy < l.threshold + 12.5)
        
        if (prevLevel !== newLevel) {
          setTransitions(t => [...t.slice(-4), { from: prevLevel, to: newLevel, time: Date.now() }])
        }
        
        return newEnergy
      })
    }, 1000)
    
    return () => clearInterval(interval)
  }, [riskLevels])

  const currentLevel = riskLevels.find(l => currentEnergy >= l.threshold && 
    currentEnergy < (l.threshold + 25)) || riskLevels[0]

  return (
    <div className="quantum-glass rounded-xl p-4">
      <h3 className="text-lg font-semibold quantum-text mb-4">Risk Quantum States</h3>
      
      {/* Energy Level Diagram */}
      <div className="relative h-48 mb-4">
        <svg className="w-full h-full">
          {/* Energy Levels */}
          {riskLevels.map((level, i) => (
            <g key={level.name}>
              <line
                x1="20%"
                y1={`${90 - level.energy * 0.8}%`}
                x2="80%"
                y2={`${90 - level.energy * 0.8}%`}
                stroke={level.color}
                strokeWidth="2"
                opacity={currentLevel.name === level.name ? 1 : 0.3}
              />
              <text
                x="10%"
                y={`${91 - level.energy * 0.8}%`}
                fill={level.color}
                fontSize="10"
                textAnchor="end"
              >
                E{i}
              </text>
              <text
                x="85%"
                y={`${91 - level.energy * 0.8}%`}
                fill={level.color}
                fontSize="10"
              >
                {level.name}
              </text>
            </g>
          ))}
          
          {/* Current State Indicator */}
          <motion.circle
            cx="50%"
            cy={`${90 - currentEnergy * 0.8}%`}
            r="6"
            fill={currentLevel.color}
            animate={{
              cy: `${90 - currentEnergy * 0.8}%`,
              fill: currentLevel.color,
            }}
            transition={{ type: "spring", stiffness: 100 }}
          >
            <animate
              attributeName="r"
              values="6;8;6"
              dur="2s"
              repeatCount="indefinite"
            />
            <animate
              attributeName="opacity"
              values="1;0.6;1"
              dur="2s"
              repeatCount="indefinite"
            />
          </motion.circle>
          
          {/* Transition Arrows */}
          {transitions.slice(-1).map((t, i) => (
            <motion.path
              key={t.time}
              d={`M 50% ${90 - riskLevels[t.from].energy * 0.8}% 
                  L 50% ${90 - riskLevels[t.to].energy * 0.8}%`}
              stroke={t.to > t.from ? '#FF0000' : '#39FF14'}
              strokeWidth="2"
              fill="none"
              initial={{ opacity: 1 }}
              animate={{ opacity: 0 }}
              transition={{ duration: 1 }}
              markerEnd="url(#arrowhead)"
            />
          ))}
          
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="10"
              refX="9"
              refY="3"
              orient="auto"
            >
              <polygon
                points="0 0, 10 3, 0 6"
                fill="currentColor"
              />
            </marker>
          </defs>
        </svg>
      </div>
      
      {/* Risk Metrics */}
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-400">Current State</span>
          <span className="text-sm font-semibold" style={{ color: currentLevel.color }}>
            {currentLevel.name}
          </span>
        </div>
        
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-400">Energy Level</span>
          <span className="text-sm font-mono text-quantum-cyan">
            {currentEnergy.toFixed(1)} eV
          </span>
        </div>
        
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-400">Transition Probability</span>
          <span className="text-sm font-mono text-quantum-execution">
            {(currentLevel.probability * 100).toFixed(0)}%
          </span>
        </div>
        
        {/* Quantum Barrier */}
        <div className="pt-3 border-t border-gray-700">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs text-gray-500">Quantum Barrier</span>
            <span className="text-xs text-quantum-energy-high">
              {(100 - currentEnergy).toFixed(0)} eV
            </span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full"
              style={{ 
                background: `linear-gradient(to right, ${currentLevel.color}, transparent)`,
                width: `${currentEnergy}%`
              }}
              animate={{ width: `${currentEnergy}%` }}
            />
          </div>
        </div>
      </div>
      
      {/* Alert for Critical State */}
      {currentEnergy > 75 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-2 rounded-lg bg-red-500/20 border border-red-500/50"
        >
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            <span className="text-xs text-red-400">Approaching Critical State</span>
          </div>
        </motion.div>
      )}
    </div>
  )
}