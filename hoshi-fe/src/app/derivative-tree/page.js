'use client';

import React, { useState, useRef } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { ChevronLeft, X, ZoomOut } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Image from 'next/image';
import { UserCircle } from 'lucide-react';
import { Stars } from '../components/derivative-tree-components/Stars';
import { Graph } from '../components/derivative-tree-components/Graph';
import hakiIcon from '../media/user-icon/haki.png';
import { useRouter } from 'next/navigation';
// import Posts from '../../../../db/posts.json';
// import Users from '../../../../db/users.json';
// import {
//   HOSHINFT_ABI,
//   HOSHINFT_CONTRACT_ADDRESS,
// } from '../../../contracts/hoshiNFT/hoshiNFT';

const createNode = (
  id,
  size,
  text,
  userName,
  userIcon,
  media = null,
  children = []
) => ({
  id,
  size,
  text,
  userName,
  userIcon,
  media,
  children,
});

const CameraController = ({ resetCamera }) => {
  const { camera, gl } = useThree();
  const controlsRef = useRef();

  React.useEffect(() => {
    resetCamera.current = () => {
      camera.position.set(0, 0, 15);
      camera.lookAt(0, 0, 0);
      controlsRef.current.reset();
    };
  }, [camera, resetCamera]);

  return (
    <OrbitControls
      ref={controlsRef}
      args={[camera, gl.domElement]}
      enablePan={true}
      enableZoom={true}
      enableRotate={true}
      minDistance={5}
      maxDistance={30}
    />
  );
};

export default function DerivativeTree() {
  const [selectedNode, setSelectedNode] = useState(null);
  const resetCamera = useRef(null);
  const router = useRouter();
  // const { primaryWallet } = useDynamicContext();

  // // using tokenId get the parents
  // // programatically add to tree

  // // function to get Image
  // const getPost = async (tokenId) => {
  //   // get post information
  //   const media = Posts[tokenId].fpath.replace('../db', '');
  //   const text = Posts[tokenId].caption;
  //   // get user information
  //   const userName = Posts[tokenId].user_handle;
  //   const userIcon = `/media/${Users[Posts[tokenId].user_handle]}`;

  //   // get children
  //   try {
  //     const walletClient = await primaryWallet.getWalletClient();
  //     const [address] = await walletClient.getAddresses();

  //     const parents = await readContract(walletClient, {
  //       address: HOSHINFT_CONTRACT_ADDRESS,
  //       abi: HOSHINFT_ABI,
  //       functionName: 'getParents',
  //       args: [address],
  //       chain: sepolia,
  //     });
  //   } catch (error) {
  //     console.error('Error interacting with the contract:', error);
  //   }
  // };

  const data = createNode(
    'root',
    0.5,
    'This is the first post but I think it is a good one. Somehow or rather ',
    'John Doe',
    hakiIcon,
    hakiIcon,
    [
      createNode(
        'child1',
        0.4,
        'Interesting point...',
        'Jane Smith',
        hakiIcon,
        hakiIcon,
        [
          createNode(
            'grandchild1',
            0.3,
            'I agree with...',
            'Bob Johnson',
            hakiIcon,
            [
              createNode(
                'greatgrandchild1',
                0.2,
                "Let's dive deeper...",
                'Alice Brown',
                hakiIcon
              ),
            ]
          ),
          createNode(
            'grandchild2',
            0.3,
            'Another perspective...',
            'Emma Wilson',
            hakiIcon
          ),
        ]
      ),
      createNode(
        'child2',
        0.4,
        'I have a different view...',
        'Michael Lee',
        hakiIcon,
        [
          createNode(
            'grandchild3',
            0.3,
            'I see what you mean...',
            'Sarah Davis',
            hakiIcon
          ),
        ]
      ),
      createNode(
        'child3',
        0.4,
        "Here's another angle...",
        'David Miller',
        hakiIcon
      ),
    ]
  );

  const handleZoomOut = () => {
    if (resetCamera.current) {
      resetCamera.current();
    }
  };

  return (
    <div className='fixed inset-x-0 top-0 bottom-16 z-50 flex flex-col bg-[#1C1C1E] text-white overflow-hidden'>
      <header className='flex justify-between items-center p-4 border-b border-gray-700'>
        <button
          className='text-purple-400'
          onClick={() => router.push('/upload/content-match')}
          aria-label='Go back'
        >
          <ChevronLeft size={24} />
        </button>
        <button
          className='text-purple-400'
          onClick={handleZoomOut}
          aria-label='Zoom out'
        >
          <ZoomOut size={24} />
        </button>
      </header>
      <Canvas
        style={{ background: '#000' }}
        camera={{ position: [0, 0, 15], fov: 60 }}
      >
        <ambientLight intensity={0.8} />
        <pointLight position={[10, 10, 10]} intensity={0.5} />
        <Stars />
        <Graph
          data={data}
          setSelectedNode={setSelectedNode}
          selectedNode={selectedNode}
        />
        <CameraController resetCamera={resetCamera} />
      </Canvas>
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            initial={{ y: '100%' }}
            animate={{ y: 0 }}
            exit={{ y: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 500 }}
            className='absolute bottom-0 left-0 right-0 bg-gray-800 p-4 z-[60] h-[40vh]'
          >
            <button
              className='absolute top-2 right-2 text-gray-400 hover:text-white'
              onClick={() => setSelectedNode(null)}
              aria-label='Close'
            >
              <X size={24} />
            </button>
            <div className='flex flex-col gap-4 mt-4'>
              <div className='flex flex-row gap-4'>
                {selectedNode.media && (
                  <Image
                    src={selectedNode.media}
                    alt='media'
                    width={180}
                    height={180}
                  />
                )}
                <p>{selectedNode.text}</p>
              </div>
              <div className='flex flex-row justify-between items-center mt-2'>
                <div className='flex items-center'>
                  {selectedNode.userIcon ? (
                    <Image
                      src={selectedNode.userIcon}
                      alt={selectedNode.userName}
                      width={32}
                      height={32}
                      className='rounded-full mr-2'
                    />
                  ) : (
                    <UserCircle size={32} className='mr-2' />
                  )}
                  <span>{selectedNode.userName}</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
