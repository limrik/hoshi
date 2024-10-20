import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import { useSpring, animated } from '@react-spring/three';
import Image from 'next/image';

export const Node = ({ node, position, setSelectedNode, selectedNode }) => {
  const meshRef = useRef(null);
  const isSelected = selectedNode && selectedNode.id === node.id;

  const { scale, color } = useSpring({
    scale: isSelected ? 1.2 : 1,
    color: isSelected ? '#ff69b4' : '#4a90e2',
    config: { tension: 300, friction: 10 },
  });

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x =
        Math.sin(state.clock.getElapsedTime() * 0.5) * 0.2;
      meshRef.current.rotation.y =
        Math.sin(state.clock.getElapsedTime() * 0.3) * 0.2;
    }
  });

  return (
    <group position={position}>
      <animated.mesh
        ref={meshRef}
        scale={scale}
        onClick={() => setSelectedNode(isSelected ? null : node)}
      >
        <sphereGeometry args={[node.size, 32, 32]} />
        <animated.meshStandardMaterial color={color} />
      </animated.mesh>
      <Html
        position={[0, node.size + 0.2, 0]}
        center
        distanceFactor={10}
        zIndexRange={[0, 0]}
      >
        <div className='bg-white bg-opacity-80 p-2 rounded-lg shadow-md text-black text-xs w-32'>
          <div className='flex items-center'>
            <Image
              src={node.userIcon}
              alt={node.userName}
              width={24}
              height={24}
              className='mr-2 rounded-full'
            />
            <span className='font-semibold'>@{node.userName}</span>
          </div>
        </div>
      </Html>
    </group>
  );
};
