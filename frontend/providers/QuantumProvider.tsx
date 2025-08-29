'use client'

import React, { createContext, useContext, useEffect, useState } from 'react'
import { create } from 'zustand'

interface QuantumState {
  marketField: number[][]
  agentStates: Map<string, AgentQuantumState>
  portfolioSuperposition: PortfolioState[]
  entanglements: EntanglementLink[]
  observerEffect: boolean
  waveFunction: WaveFunction
}

interface AgentQuantumState {
  id: string
  name: string
  energy: number
  coherence: number
  entangledWith: string[]
  state: 'idle' | 'processing' | 'deciding' | 'executing'
  personality: string
}

interface PortfolioState {
  probability: number
  positions: Map<string, number>
  expectedReturn: number
  risk: number
  collapsed: boolean
}

interface EntanglementLink {
  source: string
  target: string
  strength: number
  type: 'correlation' | 'causation' | 'quantum'
}

interface WaveFunction {
  amplitude: number[]
  phase: number[]
  coherenceTime: number
}

const useQuantumStore = create<QuantumState>((set) => ({
  marketField: Array(50).fill(null).map(() => Array(50).fill(0)),
  agentStates: new Map(),
  portfolioSuperposition: [],
  entanglements: [],
  observerEffect: false,
  waveFunction: {
    amplitude: [],
    phase: [],
    coherenceTime: 1000,
  },
}))

interface QuantumContextType {
  quantumState: QuantumState
  collapseWaveFunction: (observerId: string) => void
  entangleAgents: (agent1: string, agent2: string, strength: number) => void
  updateMarketField: (perturbation: number[][]) => void
  measurePortfolio: () => PortfolioState
}

const QuantumContext = createContext<QuantumContextType | undefined>(undefined)

export function QuantumProvider({ children }: { children: React.ReactNode }) {
  const quantumState = useQuantumStore()
  const [particles, setParticles] = useState<Array<{ x: number; y: number; vx: number; vy: number }>>([])

  useEffect(() => {
    // Initialize quantum particles for visual effects
    const newParticles = Array(20).fill(null).map(() => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 2,
      vy: (Math.random() - 0.5) * 2,
    }))
    setParticles(newParticles)

    // Animate particles
    const interval = setInterval(() => {
      setParticles(prev => prev.map(p => ({
        x: (p.x + p.vx + window.innerWidth) % window.innerWidth,
        y: (p.y + p.vy + window.innerHeight) % window.innerHeight,
        vx: p.vx + (Math.random() - 0.5) * 0.1,
        vy: p.vy + (Math.random() - 0.5) * 0.1,
      })))
    }, 50)

    return () => clearInterval(interval)
  }, [])

  const collapseWaveFunction = (observerId: string) => {
    // Simulate wave function collapse when observed
    useQuantumStore.setState(state => ({
      ...state,
      observerEffect: true,
      portfolioSuperposition: state.portfolioSuperposition.map(ps => ({
        ...ps,
        collapsed: Math.random() < ps.probability,
      })),
    }))

    setTimeout(() => {
      useQuantumStore.setState(state => ({
        ...state,
        observerEffect: false,
      }))
    }, 500)
  }

  const entangleAgents = (agent1: string, agent2: string, strength: number) => {
    useQuantumStore.setState(state => ({
      ...state,
      entanglements: [
        ...state.entanglements,
        { source: agent1, target: agent2, strength, type: 'quantum' },
      ],
    }))
  }

  const updateMarketField = (perturbation: number[][]) => {
    useQuantumStore.setState(state => {
      const newField = state.marketField.map((row, i) =>
        row.map((val, j) => val + (perturbation[i]?.[j] || 0))
      )
      return { ...state, marketField: newField }
    })
  }

  const measurePortfolio = (): PortfolioState => {
    const superposition = quantumState.portfolioSuperposition
    const totalProbability = superposition.reduce((sum, state) => sum + state.probability, 0)
    const random = Math.random() * totalProbability
    
    let cumulative = 0
    for (const state of superposition) {
      cumulative += state.probability
      if (random <= cumulative) {
        return { ...state, collapsed: true }
      }
    }
    
    return superposition[0] || {
      probability: 1,
      positions: new Map(),
      expectedReturn: 0,
      risk: 0,
      collapsed: true,
    }
  }

  return (
    <QuantumContext.Provider
      value={{
        quantumState,
        collapseWaveFunction,
        entangleAgents,
        updateMarketField,
        measurePortfolio,
      }}
    >
      {children}
      {particles.map((particle, i) => (
        <div
          key={i}
          className="quantum-particle animate-quantum-pulse"
          style={{
            left: `${particle.x}px`,
            top: `${particle.y}px`,
          }}
        />
      ))}
    </QuantumContext.Provider>
  )
}

export const useQuantum = () => {
  const context = useContext(QuantumContext)
  if (!context) {
    throw new Error('useQuantum must be used within QuantumProvider')
  }
  return context
}