import React, { useMemo } from 'react';
import * as THREE from 'three';

export const Edge = ({ start, end }) => {
  const geometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array([...start, ...end]);
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    return geometry;
  }, [start, end]);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial attach='material' color='#00ff00' linewidth={3} />
    </lineSegments>
  );
};
