import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api':  { target: 'http://localhost:8765', changeOrigin: true },
      '/ws':   { target: 'ws://localhost:8765',   changeOrigin: true, ws: true },
      '/mesh': { target: 'http://localhost:8765', changeOrigin: true },
    }
  },
  build: {
    outDir: 'dist',
    target: 'esnext',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/three')) return 'vendor-three'
          if (id.includes('node_modules/gsap')) return 'vendor-gsap'
          if (id.includes('node_modules/stats.js') || id.includes('node_modules/stats-js')) return 'vendor-stats'
          if (id.includes('node_modules/')) return 'vendor'
        }
      }
    }
  }
})
