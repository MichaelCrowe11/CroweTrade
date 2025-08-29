import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        quantum: {
          void: '#001F3F',
          blue: '#0066CC',
          cyan: '#00FFFF',
          energy: {
            high: '#8B00FF',
            medium: '#00FF00',
            low: '#FF0000',
          },
          execution: '#39FF14',
          uncertainty: 'rgba(255, 255, 255, 0.1)',
        },
      },
      animation: {
        'quantum-pulse': 'quantum-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'wave-function': 'wave-function 3s ease-in-out infinite',
        'entangle': 'entangle 4s ease-in-out infinite',
        'collapse': 'collapse 0.5s ease-out',
      },
      keyframes: {
        'quantum-pulse': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.3' },
        },
        'wave-function': {
          '0%': { transform: 'translateX(-100%) scaleY(1)' },
          '50%': { transform: 'translateX(0) scaleY(1.5)' },
          '100%': { transform: 'translateX(100%) scaleY(1)' },
        },
        'entangle': {
          '0%': { transform: 'rotate(0deg) scale(1)' },
          '50%': { transform: 'rotate(180deg) scale(1.1)' },
          '100%': { transform: 'rotate(360deg) scale(1)' },
        },
        'collapse': {
          '0%': { transform: 'scale(1.5)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}

export default config