'use client'

import React, { useRef, useEffect } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Float, MeshDistortMaterial } from '@react-three/drei'
import * as THREE from 'three'
import { useQuantum } from '@/providers/QuantumProvider'

function QuantumFieldMesh() {
  const meshRef = useRef<THREE.Mesh>(null)
  const { quantumState } = useQuantum()
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = state.clock.elapsedTime * 0.1
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.15
      
      // Distort based on market field
      const intensity = quantumState.marketField
        .flat()
        .reduce((sum, val) => sum + Math.abs(val), 0) / 2500
      
      if (meshRef.current.material && 'distort' in meshRef.current.material) {
        (meshRef.current.material as any).distort = intensity
      }
    }
  })

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
      <mesh ref={meshRef}>
        <icosahedronGeometry args={[2, 4]} />
        <MeshDistortMaterial
          color="#00FFFF"
          attach="material"
          distort={0.3}
          speed={2}
          roughness={0.2}
          metalness={0.8}
          emissive="#0066CC"
          emissiveIntensity={0.2}
        />
      </mesh>
    </Float>
  )
}

function Particles() {
  const points = useRef<THREE.Points>(null)
  const { quantumState } = useQuantum()
  
  const particlesCount = 1000
  const positions = new Float32Array(particlesCount * 3)
  
  for (let i = 0; i < particlesCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 10
    positions[i * 3 + 1] = (Math.random() - 0.5) * 10
    positions[i * 3 + 2] = (Math.random() - 0.5) * 10
  }

  useFrame((state) => {
    if (points.current) {
      points.current.rotation.y = state.clock.elapsedTime * 0.05
      points.current.rotation.x = state.clock.elapsedTime * 0.03
      
      // Pulse based on observer effect
      const scale = quantumState.observerEffect ? 1.2 : 1
      points.current.scale.lerp(new THREE.Vector3(scale, scale, scale), 0.1)
    }
  })

  return (
    <points ref={points}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particlesCount}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.02}
        color="#39FF14"
        transparent
        opacity={0.6}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}

function EntanglementLines() {
  const { quantumState } = useQuantum()
  const linesRef = useRef<THREE.Group>(null)

  useFrame((state) => {
    if (linesRef.current) {
      linesRef.current.children.forEach((child, i) => {
        if (child instanceof THREE.Line) {
          const material = child.material as THREE.LineBasicMaterial
          material.opacity = 0.3 + Math.sin(state.clock.elapsedTime * 2 + i) * 0.2
        }
      })
    }
  })

  return (
    <group ref={linesRef}>
      {quantumState.entanglements.map((link, i) => {
        const points = [
          new THREE.Vector3(Math.random() * 4 - 2, Math.random() * 4 - 2, Math.random() * 4 - 2),
          new THREE.Vector3(Math.random() * 4 - 2, Math.random() * 4 - 2, Math.random() * 4 - 2),
        ]
        const geometry = new THREE.BufferGeometry().setFromPoints(points)
        
        return (
          <line key={i}>
            <bufferGeometry attach="geometry" {...geometry} />
            <lineBasicMaterial
              attach="material"
              color="#00FFFF"
              transparent
              opacity={link.strength}
              linewidth={2}
            />
          </line>
        )
      })}
    </group>
  )
}

export default function QuantumField() {
  return (
    <div className="w-full h-full relative">
      <Canvas
        camera={{ position: [0, 0, 5], fov: 60 }}
        gl={{ antialias: true, alpha: true }}
        className="absolute inset-0"
      >
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={0.5} color="#00FFFF" />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#8B00FF" />
        
        <QuantumFieldMesh />
        <Particles />
        <EntanglementLines />
        
        <OrbitControls
          enablePan={false}
          enableZoom={true}
          maxPolarAngle={Math.PI * 0.5}
          minPolarAngle={Math.PI * 0.25}
          autoRotate
          autoRotateSpeed={0.5}
        />
        
        <fog attach="fog" args={['#001F3F', 5, 15]} />
      </Canvas>
      
      <div className="absolute bottom-4 left-4 quantum-glass px-4 py-2 rounded-lg">
        <div className="text-xs text-quantum-cyan">Quantum Field Intensity</div>
        <div className="text-lg font-mono quantum-text">
          {Math.random().toFixed(4)}
        </div>
      </div>
    </div>
  )
}