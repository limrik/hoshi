import React, { useMemo } from 'react';
import { Points, PointMaterial } from '@react-three/drei';
import { random } from 'maath';

export function Stars({ count = 5000 }) {
  const points = useMemo(() => {
    const positions = new Float32Array(count * 3);
    random.inSphere(positions, { radius: 50 });
    return positions;
  }, [count]);

  return (
    <Points positions={points}>
      <PointMaterial
        transparent
        color='#ffffff'
        size={0.05}
        sizeAttenuation={true}
        depthWrite={false}
      />
    </Points>
  );
}
