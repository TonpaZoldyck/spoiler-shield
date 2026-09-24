import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { defineConfig } from 'wxt';

/**
 * ONNX Runtime's WASM files must ship inside the extension: the Chrome Web
 * Store forbids remotely hosted code, and transformers.js would otherwise
 * fetch them from a CDN. We copy the exact build that transformers.js depends on.
 */
const ORT_FILES = ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm'];

function packageRoot(entry: string, name: string): string {
  let dir = path.dirname(fs.realpathSync(entry));
  while (dir !== path.dirname(dir)) {
    const pkg = path.join(dir, 'package.json');
    if (fs.existsSync(pkg) && JSON.parse(fs.readFileSync(pkg, 'utf8')).name === name) return dir;
    dir = path.dirname(dir);
  }
  throw new Error(`Could not find package root for ${name}`);
}

function ortDistDir(): string {
  const require = createRequire(import.meta.url);
  const transformersRoot = packageRoot(require.resolve('@huggingface/transformers'), '@huggingface/transformers');
  const fromTransformers = createRequire(path.join(transformersRoot, 'package.json'));
  return path.join(packageRoot(fromTransformers.resolve('onnxruntime-web'), 'onnxruntime-web'), 'dist');
}

export default defineConfig({
  manifest: {
    name: 'Spoiler Shield',
    description: 'Blurs film and TV spoilers as you browse, using a small model that runs on your device.',
    permissions: ['offscreen', 'storage'],
    // Cross-origin isolation gives extension pages SharedArrayBuffer, which
    // ONNX Runtime needs for multi-threaded WASM inference.
    cross_origin_embedder_policy: { value: 'require-corp' },
    cross_origin_opener_policy: { value: 'same-origin' },
    content_security_policy: {
      // 'wasm-unsafe-eval' lets extension pages compile the bundled ONNX Runtime WASM.
      extension_pages: "script-src 'self' 'wasm-unsafe-eval'; object-src 'self';",
    },
  },
  hooks: {
    'build:publicAssets': (_wxt, files) => {
      const dist = ortDistDir();
      for (const file of ORT_FILES) {
        files.push({ absoluteSrc: path.join(dist, file), relativeDest: `ort/${file}` });
      }
    },
    // Vite also emits a hashed copy of the same WASM because the ORT bundle
    // references it as a fallback. We always set wasmPaths explicitly, so that
    // copy is dead weight (27 MB). The end-to-end test proves nothing loads it.
    'build:done': (wxt) => {
      const assets = path.join(wxt.config.outDir, 'assets');
      if (!fs.existsSync(assets)) return;
      for (const name of fs.readdirSync(assets)) {
        if (/^ort-wasm-.*\.wasm$/.test(name)) fs.rmSync(path.join(assets, name));
      }
    },
  },
});
