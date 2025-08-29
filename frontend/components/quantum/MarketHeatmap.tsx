'use client'

import React, { useEffect, useState, useRef } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'

interface MarketCell {
  symbol: string
  value: number
  change: number
  volume: number
  quantumState: 'entangled' | 'coherent' | 'decoherent'
}

export default function MarketHeatmap() {
  const svgRef = useRef<SVGSVGElement>(null)
  const [marketData, setMarketData] = useState<MarketCell[]>([])
  const [selectedCell, setSelectedCell] = useState<MarketCell | null>(null)

  useEffect(() => {
    // Generate market data
    const symbols = [
      'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD',
      'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'USB', 'PNC',
      'JNJ', 'PFE', 'UNH', 'CVS', 'ABBV', 'MRK', 'TMO', 'ABT',
      'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO',
    ]
    
    const states: Array<'entangled' | 'coherent' | 'decoherent'> = ['entangled', 'coherent', 'decoherent']
    
    const data = symbols.map(symbol => ({
      symbol,
      value: Math.random() * 1000 + 100,
      change: (Math.random() - 0.5) * 10,
      volume: Math.random() * 1000000,
      quantumState: states[Math.floor(Math.random() * states.length)],
    }))
    
    setMarketData(data)
  }, [])

  useEffect(() => {
    if (!svgRef.current || marketData.length === 0) return

    const width = 900
    const height = 140
    const margin = { top: 20, right: 20, bottom: 20, left: 20 }

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)

    // Create treemap layout
    const root = d3.hierarchy({ children: marketData })
      .sum(d => d.volume)
      .sort((a, b) => b.value! - a.value!)

    const treemap = d3.treemap<typeof root>()
      .size([width - margin.left - margin.right, height - margin.top - margin.bottom])
      .padding(2)
      .round(true)

    treemap(root)

    // Color scale
    const colorScale = d3.scaleLinear<string>()
      .domain([-5, 0, 5])
      .range(['#FF0000', '#333333', '#39FF14'])

    // Create cells
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const cells = g.selectAll('g')
      .data(root.leaves())
      .enter().append('g')
      .attr('transform', d => `translate(${d.x0},${d.y0})`)

    // Add rectangles
    cells.append('rect')
      .attr('width', d => d.x1 - d.x0)
      .attr('height', d => d.y1 - d.y0)
      .attr('fill', d => colorScale(d.data.change))
      .attr('stroke', d => {
        if (d.data.quantumState === 'entangled') return '#00FFFF'
        if (d.data.quantumState === 'coherent') return '#8B00FF'
        return '#666666'
      })
      .attr('stroke-width', d => d.data.quantumState === 'entangled' ? 2 : 1)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('stroke-width', 3)
          .attr('stroke', '#00FFFF')
        
        setSelectedCell(d.data)
      })
      .on('mouseout', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('stroke-width', d.data.quantumState === 'entangled' ? 2 : 1)
          .attr('stroke', () => {
            if (d.data.quantumState === 'entangled') return '#00FFFF'
            if (d.data.quantumState === 'coherent') return '#8B00FF'
            return '#666666'
          })
        
        setSelectedCell(null)
      })

    // Add text labels
    cells.append('text')
      .attr('x', 3)
      .attr('y', 12)
      .text(d => d.data.symbol)
      .attr('font-size', d => {
        const width = d.x1 - d.x0
        return Math.min(10, width / 5) + 'px'
      })
      .attr('fill', 'white')
      .style('pointer-events', 'none')

    cells.append('text')
      .attr('x', 3)
      .attr('y', 25)
      .text(d => `${d.data.change > 0 ? '+' : ''}${d.data.change.toFixed(1)}%`)
      .attr('font-size', d => {
        const width = d.x1 - d.x0
        return Math.min(8, width / 6) + 'px'
      })
      .attr('fill', d => d.data.change > 0 ? '#39FF14' : '#FF0000')
      .style('pointer-events', 'none')

    // Animate on mount
    cells.selectAll('rect')
      .attr('opacity', 0)
      .transition()
      .duration(500)
      .delay((d, i) => i * 10)
      .attr('opacity', 1)

    // Periodic updates
    const interval = setInterval(() => {
      setMarketData(prev => prev.map(cell => ({
        ...cell,
        change: Math.max(-10, Math.min(10, cell.change + (Math.random() - 0.5) * 2)),
        volume: Math.max(100000, cell.volume + (Math.random() - 0.5) * 100000),
      })))
    }, 3000)

    return () => clearInterval(interval)
  }, [marketData])

  return (
    <div className="w-full h-full">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-lg font-semibold quantum-text">Market Quantum Field</h3>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 border-2 border-quantum-cyan rounded" />
            <span className="text-gray-400">Entangled</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 border-2 border-quantum-energy-high rounded" />
            <span className="text-gray-400">Coherent</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 border border-gray-600 rounded" />
            <span className="text-gray-400">Decoherent</span>
          </div>
        </div>
      </div>

      <div className="relative">
        <svg ref={svgRef} className="w-full" />
        
        {selectedCell && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute top-0 right-0 quantum-glass rounded-lg p-2 text-xs"
          >
            <div className="font-bold text-quantum-cyan">{selectedCell.symbol}</div>
            <div className="text-gray-400">
              Value: ${selectedCell.value.toFixed(2)}
            </div>
            <div className={selectedCell.change > 0 ? 'text-green-400' : 'text-red-400'}>
              Change: {selectedCell.change > 0 ? '+' : ''}{selectedCell.change.toFixed(2)}%
            </div>
            <div className="text-gray-400">
              Volume: {(selectedCell.volume / 1000000).toFixed(2)}M
            </div>
            <div className="text-quantum-energy-high">
              State: {selectedCell.quantumState}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}