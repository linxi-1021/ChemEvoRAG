import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import fs from 'fs'

const dataDir = path.resolve(__dirname, '..', 'data')

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'serve-data-dir',
      configureServer(server) {
        // Intercept ALL requests and check for /data/ prefix
        server.middlewares.use((req, res, next) => {
          const url = req.url || ''

          // Only handle /data/ requests
          if (!url.startsWith('/data/')) {
            return next()
          }

          // Strip /data prefix and decode
          const subPath = decodeURIComponent(url.slice('/data'.length).split('?')[0])
          const filePath = path.join(dataDir, subPath)

          // Security: prevent path traversal
          if (!filePath.startsWith(dataDir)) {
            res.writeHead(403)
            res.end('Forbidden')
            return
          }

          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            const ext = path.extname(filePath).toLowerCase()
            const contentTypes: Record<string, string> = {
              '.json': 'application/json',
              '.pdf': 'application/pdf',
              '.jpg': 'image/jpeg',
              '.jpeg': 'image/jpeg',
              '.png': 'image/png',
              '.md': 'text/plain; charset=utf-8',
              '.txt': 'text/plain; charset=utf-8',
            }
            res.setHeader('Content-Type', contentTypes[ext] || 'application/octet-stream')
            res.setHeader('Cache-Control', 'no-cache')
            fs.createReadStream(filePath).pipe(res)
          } else {
            next()
          }
        })
      },
      configurePreviewServer(server) {
        server.middlewares.use((req, res, next) => {
          const url = req.url || ''
          if (!url.startsWith('/data/')) return next()

          const subPath = decodeURIComponent(url.slice('/data'.length).split('?')[0])
          const filePath = path.join(dataDir, subPath)

          if (!filePath.startsWith(dataDir)) {
            res.writeHead(403)
            res.end('Forbidden')
            return
          }

          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            const ext = path.extname(filePath).toLowerCase()
            const contentTypes: Record<string, string> = {
              '.json': 'application/json',
              '.pdf': 'application/pdf',
              '.jpg': 'image/jpeg',
              '.md': 'text/plain; charset=utf-8',
            }
            res.setHeader('Content-Type', contentTypes[ext] || 'application/octet-stream')
            fs.createReadStream(filePath).pipe(res)
          } else {
            next()
          }
        })
      },
    },
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    fs: {
      allow: [
        path.resolve(__dirname, '..'),
      ],
    },
  },
  build: {
    rollupOptions: {
      // Don't copy public/data during build
      external: [],
    },
  },
})
