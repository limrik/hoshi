'use client';
import { useRef } from 'react';
import { useVideo } from './providers/VideoProvider';
import hoshiLogoLight from './media/logo/hoshi-logo-light.png';
import { Flame, MessageCircle } from 'lucide-react';
import Image from 'next/image';

export default function Home() {
  const openingVideo = '../media/opening-vid-hoshi.mp4';
  const { isFirstVisit, setIsFirstVisit } = useVideo();
  const videoRef = useRef(null);

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
      </div>
    </div>
  );
}
