'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Image as ImageIcon,
  X,
  Edit2,
  ChevronDown,
  TypeOutline,
  Music,
  Video,
  ChevronUp,
} from 'lucide-react';
import Image from 'next/image';
import hakiIcon from '../media/user-icon/haki.png';
import { useRouter } from 'next/navigation';
import { useUpload } from '../providers/UploadProvider';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';

export default function CreatePage() {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [selectedType, setSelectedType] = useState(null);
  const fileInputRef = useRef(null);
  const { uploadData, setUploadData } = useUpload();
  const { user } = useDynamicContext();

  const contentTypes = [
    { name: 'Text', icon: TypeOutline },
    { name: 'Image', icon: ImageIcon },
    { name: 'Sound', icon: Music },
    { name: 'Video', icon: Video },
  ];

  const router = useRouter();

  useEffect(() => {
    if (!user) {
      router.push('/');
    }
  }, [user, router]);

  const handleFileChange = (e) => {
    console.log(e.target.files[0]);
    if (e.target.files && e.target.files[0]) {
      setUploadData((prevData) => ({ ...prevData, file: e.target.files[0] }));
    }
    console.log(uploadData.content);
  };

  const handleUpload = async () => {
    setUploadData((prevData) => ({
      ...prevData,
      component: selectedType.name,
    }));
    router.push('/upload/content-match');
  };

  return (
    <div className='fixed inset-x-0 top-0 bottom-16 z-50 flex flex-col bg-[#1C1C1E] text-white overflow-hidden'>
      <AnimatePresence>
        <motion.div
          key='upload-screen'
          initial={{ x: 0 }}
          exit={{ x: '-100%' }}
          transition={{ duration: 0.3 }}
          className='flex flex-col h-full'
        >
          <header className='flex justify-between items-center p-4 border-b border-gray-700'>
            <button className='text-purple-400'>Cancel</button>
            <div className='flex justify-between items-center gap-4'>
              <input
                type='file'
                accept='image/*, video/*'
                onChange={handleFileChange}
                ref={fileInputRef}
                className='hidden'
                aria-label='Upload image'
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className='text-purple-400 hover:text-purple-300 transition-colors'
                aria-label='Add image'
              >
                <ImageIcon size={24} />
              </button>
              <button
                className={`bg-purple-600 px-4 py-1 rounded-full font-semibold ${
                  !uploadData.content && !uploadData.file
                    ? 'opacity-50 cursor-not-allowed'
                    : ''
                }`}
                onClick={handleUpload}
                disabled={!uploadData.content && !uploadData.file}
              >
                Next
              </button>
            </div>
          </header>
          <main className='flex-1 p-4 overflow-y-auto'>
            <div className='flex items-start space-x-3 mb-4'>
              <Image
                src={hakiIcon}
                alt='User avatar'
                width={40}
                height={40}
                className='rounded-full object-cover'
                unoptimized
              />
              <div className='flex-1'>
                <textarea
                  className='w-full bg-transparent text-white text-lg resize-none focus:outline-none'
                  placeholder="What's happening in the cosmos?"
                  value={uploadData.content}
                  onChange={(e) =>
                    setUploadData((prevData) => ({
                      ...prevData,
                      content: e.target.value,
                    }))
                  }
                  rows={3}
                />
              </div>
            </div>
            <AnimatePresence>
              {uploadData.file && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className='relative mt-2 rounded-xl overflow-hidden bg-gray-800'
                >
                  {/* Wrap the media in a container with a max-width */}
                  <div className='max-w-md mx-auto'>
                    {' '}
                    {/* Adjust max-width as needed */}
                    <div className='aspect-w-16 aspect-h-9'>
                      {uploadData.file.type.startsWith('video/') ? (
                        <video
                          src={URL.createObjectURL(uploadData.file)}
                          controls
                          className='w-full h-full object-cover'
                        />
                      ) : (
                        <Image
                          src={URL.createObjectURL(uploadData.file)}
                          alt='Uploaded preview'
                          width={800}
                          height={450}
                          className='w-full h-full object-cover'
                        />
                      )}
                    </div>
                  </div>
                  <div className='absolute top-2 right-2 flex space-x-2'>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className='bg-black bg-opacity-50 text-white rounded-full p-2 hover:bg-opacity-75 transition-colors'
                      aria-label='Edit image'
                    >
                      <Edit2 size={16} />
                    </button>
                    <button
                      onClick={() => setUploadData({ file: null })}
                      className='bg-black bg-opacity-50 text-white rounded-full p-2 hover:bg-opacity-75 transition-colors'
                      aria-label='Remove image'
                    >
                      <X size={16} />
                    </button>
                  </div>
                </motion.div>
              )}
              <div className='mt-4'>
                <h2 className='mb-1 font-bold mx-2'>
                  What makes your content unique?
                </h2>
                <p className='text-sm leding-tight mx-2'>
                  Choose the format that showcases your originality and value
                  the most. By standing out in the format that defines you,
                  you&apos;ll earn a higher share of IP and increase your
                  potential earnings.
                </p>
                <div className='flex justify-center mt-4'>
                  <div className='relative w-full max-w-md'>
                    <div className='flex justify-center'>
                      <button
                        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                        className={`w-3/4 bg-purple-600 text-white py-2 px-4 rounded-md hover:bg-purple-700 transition-colors flex justify-between items-center ${
                          selectedType
                            ? 'shadow-[0_0_25px_5px_rgba(168,85,247,0.7)]'
                            : !uploadData.content && !uploadData.file
                            ? 'opacity-50 cursor-not-allowed'
                            : ''
                        }`}
                        disabled={!uploadData.content && !uploadData.file}
                      >
                        {selectedType ? (
                          <div className='flex flex-row gap-2 items-center'>
                            {selectedType.icon && (
                              <selectedType.icon size={20} />
                            )}
                            {selectedType.name}
                          </div>
                        ) : (
                          'Select content type'
                        )}
                        {isDropdownOpen ? (
                          <ChevronUp size={20} />
                        ) : (
                          <ChevronDown size={20} />
                        )}
                      </button>
                    </div>
                    {isDropdownOpen && (
                      <div className='absolute top-full left-1/4 right-1/4 mt-2 bg-[#2C2C2E] rounded-lg shadow-lg w-1/2'>
                        {contentTypes.map((type) => (
                          <button
                            key={type.name}
                            onClick={() => {
                              setSelectedType(type);
                              setIsDropdownOpen(false);
                            }}
                            className='w-full text-left px-4 py-2 hover:bg-purple-600 transition-colors'
                          >
                            <div className='flex flex-row gap-2 items-center'>
                              <type.icon size={20} />
                              {type.name}
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </AnimatePresence>
          </main>
        </motion.div>
      </AnimatePresence>
      {/* {isUploading && (
        <div className='absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center'>
          <Loader className='animate-spin text-purple-400' size={48} />
        </div>
      )} */}
    </div>
  );
}
