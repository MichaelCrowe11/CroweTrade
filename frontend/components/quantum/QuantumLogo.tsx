'use client'

import React from 'react'
import { motion } from 'framer-motion'

export default function QuantumLogo({ size = 200 }: { size?: number }) {
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg
        width={size}
        height={size}
        viewBox="0 0 200 200"
        xmlns="http://www.w3.org/2000/svg"
        className="absolute inset-0"
      >
        <defs>
          <linearGradient id="quantum-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#8B00FF" stopOpacity="1" />
            <stop offset="50%" stopColor="#00FFFF" stopOpacity="1" />
            <stop offset="100%" stopColor="#39FF14" stopOpacity="1" />
          </linearGradient>
          
          <filter id="quantum-glow">
            <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>

          <pattern id="wave-pattern" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
            <path
              d="M0,20 Q10,10 20,20 T40,20"
              stroke="#00FFFF"
              strokeWidth="0.5"
              fill="none"
              opacity="0.3"
            />
          </pattern>
        </defs>

        {/* Quantum Crow Body - Composed of Wave Functions */}
        <g filter="url(#quantum-glow)">
          {/* Main Body - Probability Cloud */}
          <ellipse
            cx="100"
            cy="120"
            rx="40"
            ry="35"
            fill="url(#quantum-gradient)"
            opacity="0.8"
          />
          
          {/* Head - Collapsed Wave Function */}
          <circle
            cx="100"
            cy="80"
            r="25"
            fill="url(#quantum-gradient)"
            opacity="0.9"
          />
          
          {/* Wings - Superposition States */}
          <motion.path
            d="M60,110 Q30,100 20,120 Q30,130 50,125 Q60,120 60,110"
            fill="url(#quantum-gradient)"
            opacity="0.7"
            animate={{
              d: [
                "M60,110 Q30,100 20,120 Q30,130 50,125 Q60,120 60,110",
                "M60,110 Q25,95 15,115 Q25,125 50,125 Q60,120 60,110",
                "M60,110 Q30,100 20,120 Q30,130 50,125 Q60,120 60,110",
              ],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
          
          <motion.path
            d="M140,110 Q170,100 180,120 Q170,130 150,125 Q140,120 140,110"
            fill="url(#quantum-gradient)"
            opacity="0.7"
            animate={{
              d: [
                "M140,110 Q170,100 180,120 Q170,130 150,125 Q140,120 140,110",
                "M140,110 Q175,95 185,115 Q175,125 150,125 Q140,120 140,110",
                "M140,110 Q170,100 180,120 Q170,130 150,125 Q140,120 140,110",
              ],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut",
              delay: 0.5,
            }}
          />
          
          {/* Eye - Observer Effect */}
          <circle
            cx="100"
            cy="75"
            r="8"
            fill="#FFFFFF"
            opacity="0.9"
          />
          <motion.circle
            cx="100"
            cy="75"
            r="4"
            fill="#001F3F"
            animate={{
              r: [4, 6, 4],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
          
          {/* Beak - Quantum Measurement Tool */}
          <path
            d="M100,85 L95,92 L105,92 Z"
            fill="#39FF14"
            opacity="0.9"
          />
        </g>

        {/* Quantum Field Lines */}
        <g opacity="0.3">
          <motion.circle
            cx="100"
            cy="100"
            r="60"
            stroke="url(#quantum-gradient)"
            strokeWidth="0.5"
            fill="none"
            animate={{
              r: [60, 70, 60],
              opacity: [0.3, 0.6, 0.3],
            }}
            transition={{
              duration: 4,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
          <motion.circle
            cx="100"
            cy="100"
            r="80"
            stroke="url(#quantum-gradient)"
            strokeWidth="0.5"
            fill="none"
            animate={{
              r: [80, 90, 80],
              opacity: [0.2, 0.4, 0.2],
            }}
            transition={{
              duration: 4,
              repeat: Infinity,
              ease: "easeInOut",
              delay: 0.5,
            }}
          />
        </g>

        {/* Mathematical Symbols */}
        <text x="30" y="170" fill="#00FFFF" fontSize="12" opacity="0.6">ψ</text>
        <text x="160" y="170" fill="#8B00FF" fontSize="12" opacity="0.6">∫</text>
        <text x="100" y="180" fill="#39FF14" fontSize="10" opacity="0.5">|Ψ⟩</text>
      </svg>

      {/* Animated Particles */}
      {[...Array(6)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1 h-1 bg-quantum-cyan rounded-full"
          style={{
            left: '50%',
            top: '50%',
          }}
          animate={{
            x: [0, Math.cos(i * 60 * Math.PI / 180) * 40, 0],
            y: [0, Math.sin(i * 60 * Math.PI / 180) * 40, 0],
            opacity: [0, 1, 0],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            delay: i * 0.5,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  )
}