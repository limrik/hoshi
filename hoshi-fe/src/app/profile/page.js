'use client';

import Image from 'next/image';
import hakiIcon from '../media/user-icon/haki.png';
import { DollarSign, Sparkles } from 'lucide-react';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useDynamicContext, DynamicWidget } from '@dynamic-labs/sdk-react-core';

export default function ProfilePage() {
  const [activeTab, setActiveTab] = useState('Posts');
  const tabs = ['Posts', 'NFTs', 'Likes'];
  const router = useRouter();
  function handleClick() {
    router.push('/profile/creator-earnings-dashboard');
  }
  const { user } = useDynamicContext();
  console.log(user);
  useEffect(() => {
    if (!user) {
      router.push('/');
    }
  }, [user, router]);

  return (
    <div className='fixed top-0 bottom-16 z-50 flex flex-col bg-[#1C1C1E] text-white w-full'>
      <header className='flex justify-between items-center p-4 mb-16 w-full'>
        <div className='flex flex-col gap-4 w-full'>
          <div className='flex items-center gap-2 justify-between'>
            <div className='flex items-center gap-2'>
              <Image
                src={hakiIcon}
                alt='User avatar'
                width={50}
                height={50}
                className='rounded-full object-cover'
                unoptimized
              />
              <div className='flex flex-col'>
                <p className='text-lg font-semibold leading-tight'>
                  {user.metadata.Name}
                </p>
                <p className='text-sm text-gray-400 leading-tight'>
                  {user.metadata.Hoshihandle}
                </p>
              </div>
            </div>

            <button className='border border-gray-700 rounded-lg p-2 transition-colors duration-200 hover:bg-gray-700 hover:border-gray-600'>
              <DollarSign size={20} onClick={handleClick} />
            </button>
          </div>
          <p className='text-sm text-gray-400 leading-tight'>
            {user.metadata.Bio}
          </p>
          <div className='flex justify-center'>
            <button
              className='inline-flex items-center border border-yellow-500 rounded-lg p-2 gap-2 text-yellow-500 transition-all duration-200 hover:shadow-[0_0_10px_3px_rgba(255,255,0,0.6)] hover:border-transparent group'
              onClick={handleClick}
            >
              <Sparkles
                size={20}
                className='transition-all duration-200 group-hover:text-yellow-300 group-hover:filter group-hover:drop-shadow-[0_0_5px_rgba(255,255,0,0.6)]'
              />
              <span className='transition-all duration-200 group-hover:text-yellow-300 group-hover:filter group-hover:drop-shadow-[0_0_5px_rgba(255,255,0,0.6)]'>
                Own your own content. Cash out your IP
              </span>
            </button>
          </div>
          <div className='flex flex-row gap-4'>
            <p className='text-sm leading-tight'>
              790 <span className='text-gray-400 '>following</span>
            </p>
            <p className='text-sm leading-tight'>
              410k <span className='text-gray-400 '>followers</span>
            </p>
          </div>
          <DynamicWidget />
        </div>
      </header>
      <div className='flex border-b border-gray-700'>
        {tabs.map((tab) => (
          <button
            key={tab}
            className={`flex-1 py-2 text-sm font-medium ${
              activeTab === tab
                ? 'text-white border-b-2 border-yellow-500'
                : 'text-gray-400'
            }`}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </div>

      <main className='flex-1 overflow-y-auto p-4'>
        {activeTab === 'Posts' && (
          <>
            <div className='grid grid-cols-3 gap-1'>
              {[...Array(9)].map((_, index) => (
                <div
                  key={index}
                  className='aspect-square bg-gray-700 rounded-md overflow-hidden relative group'
                >
                  <Image
                    src={hakiIcon}
                    alt={`Post ${index + 1}`}
                    width={400}
                    height={400}
                    className='w-full h-full object-cover'
                  />
                  <div className='absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200'>
                    <p className='text-white text-sm'>View Post</p>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
        {activeTab === 'NFTs' && (
          <div className='text-center py-8'>
            <p>User has no NFTs</p>
          </div>
        )}
        {activeTab === 'Likes' && (
          <div className='text-center py-8'>
            <p>No liked posts</p>
          </div>
        )}
      </main>
    </div>
  );
}
