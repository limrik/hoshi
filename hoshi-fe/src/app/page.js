'use client';
import { useState, useRef, useEffect } from 'react';
import { useVideo } from './providers/VideoProvider';
import hoshiLogoLight from './media/logo/hoshi-logo-light.png';
import { Ellipsis, Flame, MessageCircle, Network } from 'lucide-react';
import Image from 'next/image';
import { motion } from 'framer-motion';
import { FaRegStar, FaStar } from 'react-icons/fa';
import Drawer from './components/Drawer';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import LoginPage from './components/LoginPage';
import Users from '../../public/db/users.json';
import { readContract, writeContract } from 'viem/actions';
import {
  HOSHITOKEN_ABI,
  HOSHITOKEN_CONTRACT_ADDRESS,
} from '../../contracts/hoshitoken/hoshitoken';
import { flowTestnet } from 'viem/chains';
import hoshitoken from './media/logo/token-icon.png';
import { formatUnits } from 'viem';

export default function Home() {
  const { user } = useDynamicContext();

  const openingVideo = '../media/opening-vid-hoshi.mp4';
  const { isFirstVisit, setIsFirstVisit } = useVideo();
  const videoRef = useRef(null);

  const [selectedPost, setSelectedPost] = useState(null);
  const openDrawer = (index) => setSelectedPost(index);
  const closeDrawer = () => setSelectedPost(null);

  const [posts, setPosts] = useState('');
  const { primaryWallet } = useDynamicContext();
  const [tokens, setTokens] = useState();

  async function getPosts() {
    try {
      const response = await fetch(
        'https://e3c4-104-244-25-79.ngrok-free.app/posts/',
        {
          method: 'GET',
          headers: {
            'ngrok-skip-browser-warning': '69420',
          },
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Error Response:', errorData);
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Success:', data);
      return data;
    } catch (error) {
      console.error('Error:', error);
      return error;
    }
  }

  useEffect(() => {
    getPosts().then((data) => {
      setPosts(data);
    });
    const getHoshitokens = async () => {
      try {
        const walletClient = await primaryWallet.getWalletClient();
        const [address] = await walletClient.getAddresses();

        const hoshitokens = await readContract(walletClient, {
          address: HOSHITOKEN_CONTRACT_ADDRESS,
          abi: HOSHITOKEN_ABI,
          functionName: 'balanceOf',
          args: [address],
          chain: flowTestnet,
        });
        console.log(hoshitokens);
        setTokens(formatUnits(hoshitokens, 18));
      } catch (error) {
        console.error('Error interacting with the contract:', error);
      }
    };

    getHoshitokens();
  }, [primaryWallet]);

  const handleLikeClick = async (id) => {
    // interact with smart contract
    // console.log(id);
    const handleLike = async () => {
      try {
        const walletClient = await primaryWallet.getWalletClient();

        const tx = await writeContract(walletClient, {
          address: HOSHITOKEN_CONTRACT_ADDRESS,
          abi: HOSHITOKEN_ABI,
          functionName: 'likePost',
          args: [id, 100 * 10 ** 18],
          chain: flowTestnet,
        });
        console.log('Subscription transaction sent:', tx);

        const receipt = await tx.wait();
        console.log('Transaction confirmed:', receipt);
      } catch (error) {
        console.error('Error interacting with the contract:', error);
      }
    };
    handleLike();

    // const handleLike = async () => {
    //   try {
    //     const walletClient = await primaryWallet.getWalletClient();

    //     const tx = await writeContract(walletClient, {
    //       address: HOSHINFT_CONTRACT_ADDRESS,
    //       abi: HOSHINFT_ABI,
    //       functionName: 'mintNFT',
    //       args: [
    //         '0xFa5530BE79c0dce48De0Da80a1A11Bf8465B99d9',
    //         [],
    //         [],
    //         'https://tomato-occupational-carp-281.mypinata.cloud/ipfs/QmRZjD9NPqi153TEz8fpXfyiyt3YSBNEjWxqnJPHRy2Bmc',
    //       ],
    //       chain: sepolia,
    //     });
    //     console.log('Subscription transaction sent:', tx);

    //     const receipt = await tx.wait();

    //     const transferEvent = receipt.events.find(
    //       (event) => event.event === 'NFTMinted'
    //     );

    //     // Extract the token ID (3rd argument in the Transfer event)
    //     const tokenId = transferEvent.args[2].toNumber(); // Assuming tokenId is a uint256
    //     console.log('Minted Token ID:', tokenId);
    //     console.log('Transaction confirmed:', receipt);
    //   } catch (error) {
    //     console.error('Error interacting with the contract:', error);
    //   }
    // };
    // handleLike();

    setPosts((prevPosts) =>
      prevPosts.map((post) =>
        post.token_id === id
          ? {
              ...post,
              liked: true,
              liked_count: post.liked ? post.liked_count : post.liked_count + 1,
            }
          : post
      )
    );
  };

  return (
    <div
      className={`relative min-h-screen w-screen bg-black ${
        isFirstVisit && 'no-scroll'
      }`}
    >
      {/* Opening Video */}
      {isFirstVisit && (
        <video
          ref={videoRef}
          className='fixed top-0 left-0 w-full h-full object-cover transition-opacity duration-1000 z-50'
          src={openingVideo}
          playsInline
          muted
          autoPlay
          onEnded={() => setIsFirstVisit(false)}
        >
          Your browser does not support the video tag.
        </video>
      )}
      {!isFirstVisit && user && (
        <div className='relative min-h-screen flex flex-col bg-[#1C1C1E] text-white'>
          <header className='z-50 fixed top-0 inset-x-0 h-16 flex justify-center items-center p-4 border-b border-gray-700 bg-[#1C1C1E]'>
            <Image
              src={hoshiLogoLight}
              alt='Hoshi logo'
              width={100}
              height={140}
              unoptimized
            />
            <div className='absolute right-8 flex flex-row gap-1'>
              <div className='text-[#fff4d1]'>{tokens}</div>
              <Image
                src={hoshitoken}
                alt='Hoshi token'
                width={20}
                height={20}
              />
            </div>
          </header>
          <main className='mt-16 flex-1 overflow-y-auto z-20'>
            {posts &&
              posts.map((post, id) => {
                const updatedFpath = post.fpath.replace(
                  '../db/media',
                  '/db/media'
                );
                const user = Users.find(
                  (user) => user.hoshiHandle === post.user_handle
                );
                const userAvatar = `/media/${user.imagePath}`;

                return (
                  <div key={id} className='p-8'>
                    <div className='rounded-t-lg bg-gray-800 text-white p-4 flex items-center gap-2'>
                      <Image
                        src={userAvatar}
                        alt='User avatar'
                        width={40}
                        height={40}
                        className='rounded-full'
                        unoptimized
                      />
                      <div className='text-sm'>{post.user_handle}</div>
                      <button
                        onClick={() => openDrawer(id)}
                        className='ml-auto flex border border-gray-700 rounded-full p-1 hover:bg-gray-700 hover:border-gray-600 border-opacity-10'
                      >
                        <Ellipsis size={25} color='#fff4d1' />
                      </button>
                    </div>
                    <div className='relative overflow-hidden'>
                      <div className='aspect-w-16 aspect-h-9'>
                        <Image
                          src={updatedFpath}
                          alt='Uploaded preview'
                          width={800}
                          height={450}
                          className='w-full h-full object-cover'
                        />
                      </div>
                    </div>
                    <div className='flex items-center justify-between bg-gray-800 px-4 pb-2 pt-2 rounded-b-lg'>
                      <p className='text-xs'>
                        <span className=' font-bold mr-2 '>
                          {post.user_handle}
                        </span>
                        {post.caption}
                      </p>
                      <div className='relative inline-block '>
                        <motion.div
                          whileTap={{ scale: 0.8 }}
                          onClick={() => handleLikeClick(post.token_id)}
                          style={{ cursor: 'pointer' }}
                          aria-pressed={post.liked}
                          aria-label={post.liked ? 'Unlike' : 'Like'}
                          className='flex items-center bg-gray-800 border border-gray-400 text-white rounded-full py-1.5 px-2 shadow-lg border-opacity-35 hover:border-opacity-100'
                        >
                          {post.liked ? (
                            <FaStar
                              size={12}
                              className='mr-3'
                              color='#fff4d1'
                            />
                          ) : (
                            <FaRegStar
                              size={12}
                              className='mr-3'
                              color='#fff4d1'
                            />
                          )}
                          <span
                            className={`text-xs  ${
                              post.liked ? 'font-bold' : ''
                            }`}
                          >
                            {post.liked_count}
                          </span>
                        </motion.div>
                      </div>
                    </div>
                  </div>
                );
              })}

            {/* Drawer */}
            <Drawer isOpen={selectedPost !== null} onClose={closeDrawer}>
              <div className='flex flex-col p-4 bg-gray-800 border border-gray-700 hover:bg-gray-700 rounded-lg'>
                <button className='flex flex-row gap-4'>
                  <Network size={20} color='gray' />
                  <div className='text-gray-200'>View derivative tree</div>
                </button>
              </div>
              <button
                onClick={closeDrawer}
                className='mt-4 p-2 bg-red-600 rounded-md hover:bg-red-500'
              >
                Close
              </button>
            </Drawer>
          </main>
        </div>
      )}

      {/* logged out view */}
      {!isFirstVisit && !user && <LoginPage />}
    </div>
  );
}
