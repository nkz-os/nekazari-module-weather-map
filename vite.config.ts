import { defineConfig } from 'vite';
import { nkzModulePreset } from '@nekazari/module-builder';

export default defineConfig({
  presets: [nkzModulePreset()],
});
