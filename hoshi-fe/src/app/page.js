'use client';
import { useState, useRef } from 'react';
import { useVideo } from './providers/VideoProvider';
import hoshiLogoLight from './media/logo/hoshi-logo-light.png';
import { Ellipsis, Flame, MessageCircle, Network } from 'lucide-react';
import Image from 'next/image';
import hakiIcon from './media/user-icon/haki.png';
import { motion } from 'framer-motion';
import { FaRegStar, FaStar } from 'react-icons/fa';
import Drawer from './components/Drawer';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import LoginPage from './components/LoginPage';

export default function Home() {
  const { user } = useDynamicContext();

  const openingVideo = '../media/opening-vid-hoshi.mp4';
  const { isFirstVisit, setIsFirstVisit } = useVideo();
  const videoRef = useRef(null);

  const [selectedPost, setSelectedPost] = useState(null);
  const openDrawer = (index) => setSelectedPost(index);
  const closeDrawer = () => setSelectedPost(null);

  const [posts, setPosts] = useState([
    { id: 1, content: 'This is a post', liked: false, likesCount: 10 },
    { id: 2, content: 'Another post content', liked: false, likesCount: 5 },
    { id: 3, content: 'Yet another post', liked: false, likesCount: 7 },
  ]);

  const handleLikeClick = (id) => {
    setPosts((prevPosts) =>
      prevPosts.map((post) =>
        post.id === id
          ? {
              ...post,
              liked: !post.liked,
              likesCount: post.liked
                ? post.likesCount - 1
                : post.likesCount + 1,
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
          className='fixed top-0 left-0 w-full h-full object-cover transition-opacity duration-1000'
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
            <div className='absolute right-4 flex flex-row gap-1'>
              <button className='border border-gray-700 rounded-lg p-2 hover:bg-gray-700 hover:border-gray-600'>
                <Flame size={20} color='#fff4d1' />
              </button>
              <button className='border border-gray-700 rounded-lg p-2 hover:bg-gray-700 hover:border-gray-600'>
                <MessageCircle size={20} color='#fff4d1' />
              </button>
            </div>
          </header>
          <main className='mt-16 flex-1 overflow-y-auto z-20'>
            {posts.map((post) => (
              <div key={post.id} className='p-8'>
                <div className='rounded-t-lg bg-gray-800 text-white p-4 flex items-center gap-2'>
                  <Image
                    src={hakiIcon}
                    alt='User avatar'
                    width={40}
                    height={40}
                    className='rounded-full'
                    unoptimized
                  />
                  <div>haki</div>
                  <button
                    onClick={() => openDrawer(post.id)}
                    className='ml-auto flex border border-gray-700 rounded-lg p-2 hover:bg-gray-700 hover:border-gray-600'
                  >
                    <Ellipsis size={20} color='#fff4d1' />
                  </button>
                </div>
                <div className='relative overflow-hidden'>
                  <div className='aspect-w-16 aspect-h-9'>
                    <Image
                      src={hakiIcon}
                      alt='Uploaded preview'
                      width={800}
                      height={450}
                      className='w-full h-full object-cover'
                    />
                  </div>
                </div>
                <div className='flex items-center justify-between bg-gray-800 p-4 rounded-b-lg'>
                  <p>{post.content}</p>
                  <div className='relative inline-block'>
                    <motion.div
                      whileTap={{ scale: 0.8 }}
                      onClick={() => handleLikeClick(post.id)}
                      style={{ cursor: 'pointer' }}
                      aria-pressed={post.liked}
                      aria-label={post.liked ? 'Unlike' : 'Like'}
                      className='flex items-center bg-gray-800 border border-gray-400 text-white rounded-full py-1 px-2 shadow-lg'
                    >
                      {post.liked ? (
                        <FaStar size={16} className='mr-1' color='#fff4d1' />
                      ) : (
                        <FaRegStar size={16} className='mr-1' color='#fff4d1' />
                      )}
                      <span className='text-sm font-bold'>
                        {post.likesCount}
                      </span>
                    </motion.div>
                  </div>
                </div>
              </div>
            ))}

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
