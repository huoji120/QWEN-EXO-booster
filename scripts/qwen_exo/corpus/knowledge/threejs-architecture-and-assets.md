---
canonical: true
quality: 0.98
source_kind: threejs_production_reference
---
# Three.js Architecture and Asset Pipeline Reference

This reference covers the production Three.js execution contract, renderer and scene ownership, camera discipline, GLB/glTF loading, texture compression, environment setup, and a complete framework-neutral baseline lifecycle.

## 0. Core execution contract

When asked to build or improve a Three.js experience:

1. Inspect the existing package manager, framework, Three.js version, renderer setup, asset paths, and component lifecycle before editing. Reuse the project's architecture; do not introduce React Three Fiber, a second render loop, another state system, or a CDN build beside an existing module build.
2. Deliver a runnable implementation, not a concept sketch. A scene must load, resize, render, respond to input, show loading/error states, and release GPU resources when removed.
3. Start with one `Scene`, one primary `Camera`, one renderer, one owned canvas, and one explicit lifecycle. Add abstractions only when repeated behavior justifies them.
4. Use `WebGLRenderer` as the conservative production default. Do not switch to WebGPU unless the project already targets it and the required browser/device support is verified.
5. Preserve visual intent while enforcing a frame budget. Measure draw calls, triangles, texture memory, frame time, loading time, and device pixel ratio. Never claim an optimization without observing the relevant metric.
6. Prefer GLB/glTF for runtime assets. Compress geometry with Draco or Meshopt when the deployed loader supports it; compress textures with KTX2/Basis when the hosting pipeline and target GPUs are verified.
7. Share geometry, materials, and textures. Use `InstancedMesh` for repeated objects and merge static geometry only when it does not break culling, materials, interaction, or updates.
8. Avoid per-frame allocations. Reuse `Vector2`, `Vector3`, `Quaternion`, `Matrix4`, colors, raycast arrays, and temporary objects. Do not create materials, geometries, loaders, or DOM nodes inside the animation loop.
9. Cap render resolution. Derive the drawing buffer from the canvas display size and a bounded device pixel ratio; high-DPI screens must not silently multiply fragment cost by 4–9×.
10. Keep lighting and shadows bounded. Start with environment lighting plus one intentional key light. Limit shadow-casting lights, shadow-map size, shadow camera bounds, and the number of casting/receiving meshes.
11. Render on demand for mostly static viewers and editors. Use continuous animation only while objects, controls, video textures, simulations, or effects are changing.
12. Dispose owned geometries, materials, textures, render targets, controls, loaders/workers where supported, event listeners, animation frames, and observers. Removing a canvas from the DOM is not GPU cleanup.
13. Provide a non-3D fallback or useful loading/error copy. Respect `prefers-reduced-motion`, keyboard access, touch input, and readable HTML outside the canvas.
14. Verify on narrow, medium, and wide layouts, at least one lower-power profile, and browser resize/visibility transitions. The final check is the running scene, not source inspection.

## 1. Production baseline structure

Keep ownership explicit. The following module pattern is intentionally small and framework-neutral:

```js
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function createThreeApp(canvas) {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
  camera.position.set(3, 2, 5);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.target.set(0, 0.8, 0);

  const clock = new THREE.Clock();
  const disposables = new Set();
  let frame = 0;
  let running = true;

  function track(resource) {
    if (resource?.dispose) disposables.add(resource);
    return resource;
  }

  function resize() {
    const dprCap = matchMedia('(max-width: 700px)').matches ? 1.25 : 1.5;
    const dpr = Math.min(window.devicePixelRatio || 1, dprCap);
    const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width === width && canvas.height === height) return false;
    renderer.setSize(width, height, false);
    camera.aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
    camera.updateProjectionMatrix();
    return true;
  }

  function animate() {
    if (!running) return;
    const dt = Math.min(clock.getDelta(), 0.05);
    resize();
    controls.update(dt);
    renderer.render(scene, camera);
    frame = requestAnimationFrame(animate);
  }

  const onVisibility = () => {
    clock.getDelta();
    if (!document.hidden && running && !frame) animate();
  };
  document.addEventListener('visibilitychange', onVisibility);
  animate();

  return {
    scene,
    camera,
    renderer,
    controls,
    track,
    dispose() {
      running = false;
      cancelAnimationFrame(frame);
      frame = 0;
      document.removeEventListener('visibilitychange', onVisibility);
      controls.dispose();
      scene.traverse((object) => {
        object.geometry?.dispose?.();
        const materials = Array.isArray(object.material)
          ? object.material
          : object.material ? [object.material] : [];
        for (const material of materials) {
          for (const value of Object.values(material)) {
            if (value?.isTexture) value.dispose();
          }
          material.dispose();
        }
      });
      for (const resource of disposables) resource.dispose();
      disposables.clear();
      renderer.dispose();
      renderer.forceContextLoss?.();
    },
  };
}
```

Adapt the ownership boundary to React/Vue/Svelte lifecycle hooks, but preserve one initialization and one cleanup. In React development Strict Mode, initialization may run twice; cleanup must be complete and idempotent.

## 2. Scene, camera, and coordinate discipline

- Define a world scale before importing assets. A useful default is one world unit equals one meter. Normalize imported model scale once rather than compensating throughout camera and lighting code.
- Keep the model near the origin unless large-world coordinates are a requirement. Very large camera near/far ratios reduce depth precision and cause z-fighting.
- Fit the camera from a `Box3` only after the model has loaded and world matrices are updated. Compute the bounding sphere or box, set the controls target, derive camera distance from vertical and horizontal FOV, and update clipping planes conservatively.
- Use `camera.lookAt` or controls target deliberately. Do not fight OrbitControls by overwriting camera transforms every frame.
- Keep static objects' transforms static. For large static subtrees, set `matrixAutoUpdate = false` only after setting and updating matrices correctly.
- Use layers for selective rendering/raycasting when UI helpers, effects, or picking targets differ from the visible scene.

## 3. Asset pipeline: GLB, textures, and environment

Prefer `.glb` for production delivery because geometry, animation, and material references travel together. Load through `GLTFLoader`, provide explicit progress/error UI, and cancel or ignore stale loads when a component unmounts.

```js
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';

const manager = new THREE.LoadingManager();
const loader = new GLTFLoader(manager);

const draco = new DRACOLoader(manager);
draco.setDecoderPath('/draco/');
loader.setDRACOLoader(draco);

const ktx2 = new KTX2Loader(manager)
  .setTranscoderPath('/basis/')
  .detectSupport(renderer);
loader.setKTX2Loader(ktx2);

const gltf = await loader.loadAsync('/models/product.glb');
scene.add(gltf.scene);
```

Operational rules:

- Verify decoder/transcoder assets are deployed at the configured paths. A locally cached decoder is not deployment proof.
- Use Meshopt only when the asset was encoded with Meshopt and `GLTFLoader` has the decoder configured.
- Resize source textures before delivery. A 4K texture remains expensive in GPU memory even when its downloaded JPEG is small.
- Use KTX2/Basis for GPU-compressed color/normal maps where visual quality is acceptable. Test target devices; compression formats and transcoding paths vary.
- Set color textures such as base color, emissive, and UI sprites to `SRGBColorSpace` when they are not already configured by the glTF loader. Keep normal, roughness, metalness, AO, depth, and data textures in linear/no-color space.
- Reuse textures and materials. Cloning an entire loaded scene can duplicate mutable material state and increase memory; decide whether clones share or own resources.
- Use an HDR environment through `PMREMGenerator` for physically based materials, then dispose the source HDR texture and temporary generator when no longer needed.
- Treat asset failure as a visible state. Show what failed and provide retry/fallback behavior instead of leaving an empty canvas.
