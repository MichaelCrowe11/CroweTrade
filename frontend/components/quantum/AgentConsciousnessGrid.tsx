'use client'

import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuantum } from '@/providers/QuantumProvider'

interface AgentNode {
  id: string
  name: string
  personality: string
  energy: number
  coherence: number
  state: 'idle' | 'processing' | 'deciding' | 'executing'
  position: { x: number; y: number }
  connections: string[]
}

const AGENT_PERSONALITIES = [
  { id: 'signal-1', name: 'Alpha Scanner', personality: 'Aggressive momentum hunter', specialty: 'breakouts' },
  { id: 'signal-2', name: 'Theta Analyst', personality: 'Mean reversion specialist', specialty: 'reversals' },
  { id: 'signal-3', name: 'Gamma Predictor', personality: 'Pattern recognition expert', specialty: 'fractals' },
  { id: 'portfolio-1', name: 'Risk Guardian', personality: 'Conservative protector', specialty: 'drawdown control' },
  { id: 'portfolio-2', name: 'Allocator Prime', personality: 'Dynamic optimizer', specialty: 'position sizing' },
  { id: 'portfolio-3', name: 'Correlation Matrix', personality: 'Relationship mapper', specialty: 'diversification' },
  { id: 'execution-1', name: 'Speed Daemon', personality: 'Latency minimizer', specialty: 'HFT execution' },
  { id: 'execution-2', name: 'Stealth Router', personality: 'Market impact reducer', specialty: 'dark pools' },
  { id: 'risk-1', name: 'Sentinel VAR', personality: 'Value-at-risk calculator', specialty: 'tail risk' },
  { id: 'risk-2', name: 'Regime Detector', personality: 'Market state identifier', specialty: 'volatility regimes' },
  { id: 'ml-1', name: 'Neural Prophet', personality: 'Deep learning oracle', specialty: 'price prediction' },
  { id: 'ml-2', name: 'Feature Engineer', personality: 'Signal transformer', specialty: 'feature extraction' },
]

function AgentAvatar({ agent }: { agent: AgentNode }) {
  const stateColors = {
    idle: 'border-gray-600',
    processing: 'border-quantum-cyan',
    deciding: 'border-yellow-500',
    executing: 'border-quantum-execution',
  }

  const energyLevel = Math.min(100, Math.max(0, agent.energy))
  const pulseIntensity = agent.state === 'executing' ? 2 : agent.state === 'processing' ? 1.5 : 1

  return (
    <motion.div
      className={`relative w-24 h-24 rounded-full border-2 ${stateColors[agent.state]} 
        quantum-glass cursor-pointer group`}
      initial={{ scale: 0, opacity: 0 }}
      animate={{ 
        scale: 1, 
        opacity: 1,
        boxShadow: `0 0 ${20 * pulseIntensity}px rgba(0, 255, 255, ${0.3 * pulseIntensity})`
      }}
      whileHover={{ scale: 1.1 }}
      transition={{ duration: 0.3 }}
    >
      {/* Energy Core */}
      <div 
        className="absolute inset-2 rounded-full bg-gradient-radial from-quantum-cyan to-transparent"
        style={{ 
          opacity: energyLevel / 100,
          filter: `blur(${2 - (energyLevel / 50)}px)`,
        }}
      />
      
      {/* Coherence Ring */}
      <svg className="absolute inset-0 w-full h-full -rotate-90">
        <circle
          cx="48"
          cy="48"
          r="44"
          stroke="url(#gradient)"
          strokeWidth="2"
          fill="none"
          strokeDasharray={`${agent.coherence * 2.76} 276`}
          className="transition-all duration-500"
        />
        <defs>
          <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#8B00FF" />
            <stop offset="50%" stopColor="#00FFFF" />
            <stop offset="100%" stopColor="#39FF14" />
          </linearGradient>
        </defs>
      </svg>

      {/* Agent Icon */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <div className="text-2xl mb-1">
            {agent.state === 'idle' ? '🧠' : 
             agent.state === 'processing' ? '⚡' :
             agent.state === 'deciding' ? '🎯' : '🚀'}
          </div>
        </div>
      </div>

      {/* Tooltip */}
      <div className="absolute -bottom-20 left-1/2 -translate-x-1/2 w-48 
        opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none">
        <div className="quantum-glass rounded-lg p-2 text-xs">
          <div className="font-bold text-quantum-cyan">{agent.name}</div>
          <div className="text-gray-400">{agent.personality}</div>
          <div className="mt-1 flex justify-between">
            <span>Energy: {energyLevel}%</span>
            <span>Coherence: {agent.coherence}%</span>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function EntanglementWeb({ agents }: { agents: AgentNode[] }) {
  const { quantumState } = useQuantum()
  
  return (
    <svg className="absolute inset-0 pointer-events-none">
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>
      
      {quantumState.entanglements.map((link, i) => {
        const source = agents.find(a => a.id === link.source)
        const target = agents.find(a => a.id === link.target)
        
        if (!source || !target) return null
        
        return (
          <motion.line
            key={i}
            x1={source.position.x + 48}
            y1={source.position.y + 48}
            x2={target.position.x + 48}
            y2={target.position.y + 48}
            stroke="#00FFFF"
            strokeWidth={link.strength * 2}
            opacity={link.strength}
            filter="url(#glow)"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, ease: "easeInOut" }}
          />
        )
      })}
    </svg>
  )
}

export default function AgentConsciousnessGrid() {
  const [agents, setAgents] = useState<AgentNode[]>([])
  const { entangleAgents } = useQuantum()

  useEffect(() => {
    // Initialize agents with positions
    const gridSize = 4
    const spacing = 150
    const startX = 50
    const startY = 50
    
    const initialAgents = AGENT_PERSONALITIES.slice(0, 12).map((personality, i) => ({
      ...personality,
      energy: 50 + Math.random() * 50,
      coherence: 60 + Math.random() * 40,
      state: 'idle' as const,
      position: {
        x: startX + (i % gridSize) * spacing,
        y: startY + Math.floor(i / gridSize) * spacing,
      },
      connections: [],
    }))
    
    setAgents(initialAgents)
    
    // Simulate agent activity
    const interval = setInterval(() => {
      setAgents(prev => prev.map(agent => {
        const states: Array<'idle' | 'processing' | 'deciding' | 'executing'> = 
          ['idle', 'processing', 'deciding', 'executing']
        const newState = Math.random() > 0.7 ? 
          states[Math.floor(Math.random() * states.length)] : agent.state
        
        return {
          ...agent,
          state: newState,
          energy: Math.max(20, Math.min(100, agent.energy + (Math.random() - 0.5) * 10)),
          coherence: Math.max(30, Math.min(100, agent.coherence + (Math.random() - 0.5) * 5)),
        }
      }))
      
      // Random entanglements
      if (Math.random() > 0.8) {
        const agent1 = AGENT_PERSONALITIES[Math.floor(Math.random() * 12)]
        const agent2 = AGENT_PERSONALITIES[Math.floor(Math.random() * 12)]
        if (agent1.id !== agent2.id) {
          entangleAgents(agent1.id, agent2.id, Math.random())
        }
      }
    }, 2000)
    
    return () => clearInterval(interval)
  }, [entangleAgents])

  return (
    <div className="relative w-full h-[600px] quantum-glass rounded-xl p-6 overflow-hidden">
      <div className="absolute top-4 left-4 z-10">
        <h2 className="text-2xl font-bold quantum-text mb-2">Agent Consciousness Grid</h2>
        <p className="text-sm text-gray-400">24 PhD-level quantum minds orchestrating trades</p>
      </div>
      
      <div className="relative w-full h-full mt-12">
        <EntanglementWeb agents={agents} />
        
        <AnimatePresence>
          {agents.map(agent => (
            <div
              key={agent.id}
              className="absolute"
              style={{ left: agent.position.x, top: agent.position.y }}
            >
              <AgentAvatar agent={agent} />
            </div>
          ))}
        </AnimatePresence>
      </div>
      
      {/* Status Bar */}
      <div className="absolute bottom-4 left-4 right-4 flex justify-between text-xs">
        <div className="flex gap-4">
          <span className="text-gray-400">Active Agents: </span>
          <span className="text-quantum-cyan">
            {agents.filter(a => a.state !== 'idle').length}/{agents.length}
          </span>
        </div>
        <div className="flex gap-4">
          <span className="text-gray-400">Avg Coherence: </span>
          <span className="text-quantum-execution">
            {(agents.reduce((sum, a) => sum + a.coherence, 0) / agents.length).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  )
}