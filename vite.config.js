import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import basicSsl from '@vitejs/plugin-basic-ssl';

export default defineConfig({
  plugins: [
    react(),
    basicSsl()
  ],
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    https: true
  }
});
