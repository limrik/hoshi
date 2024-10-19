'use client';
import { useUpload } from '../providers/UploadProvider';
import { useState } from 'react';
import Image from 'next/image';
import hakiIcon from '../media/user-icon/haki.png';
import { useRouter } from 'next/navigation'; // Import useRouter for navigation
import { ChevronLeft } from 'lucide-react';

export default function DisputePage() {
  const [disputeReason, setDisputeReason] = useState('');
  const [disputeSubmitted, setDisputeSubmitted] = useState(false); // New state variable
  const router = useRouter(); // Initialize useRouter

  const { uploadData } = useUpload();

  const handleDisputeSubmit = (e) => {
    e.preventDefault();
    console.log('Dispute Reason:', disputeReason);
    setDisputeSubmitted(true); // Update state to show the submission screen

    // Redirect to "/" after a delay
    setTimeout(() => {
      router.push('/');
    }, 3000); // Redirect after 3 seconds
  };

  // Conditional rendering based on disputeSubmitted state
  if (disputeSubmitted) {
    return (
      <div className='fixed inset-x-0 top-0 bottom-16 z-50 flex flex-col bg-[#1C1C1E] text-white overflow-hidden'>
        <main className='flex flex-col justify-center items-center m-10 h-full'>
          <h1 className='text-2xl font-bold'>Dispute Submitted</h1>
          <p className='text-sm text-gray-400 text-center'>
            Thank you for submitting your dispute. Our team will review your
            case, and you will receive a notification with the outcome within
            3-5 business days. You will be redirected shortly.
          </p>
        </main>
      </div>
    );
  }

  return (
    <div className='fixed inset-x-0 top-0 bottom-16 z-50 flex flex-col bg-[#1C1C1E] text-white'>
      <header className='flex justify-between items-center p-4 border-b border-gray-700'>
        <button
          className='text-purple-400'
          onClick={() => router.back()}
          aria-label='Go back'
        >
          <ChevronLeft size={24} />
        </button>
      </header>
      <main className='flex-1 flex flex-col overflow-y-auto m-10'>
        <div className='relative flex flex-col h-[60vh] justify-between'>
          <div className='flex flex-row'>
            {uploadData.file && (
              <div className='w-1/2 relative overflow-hidden rounded-l-lg'>
                <div className='aspect-w-16 aspect-h-9'>
                  <Image
                    src={URL.createObjectURL(uploadData.file)}
                    alt='Uploaded preview'
                    width={400}
                    height={225}
                    className='w-full h-full object-cover'
                  />
                </div>
              </div>
            )}
            {uploadData.content && (
              <div className='w-1/2 p-4 flex bg-gray-800 rounded-r-lg relative'>
                <p className='whitespace-pre-wrap'>{uploadData.content}</p>
                <div className='absolute translate-y-1/4 bottom-0 right-2 flex items-center bg-gray-700 rounded-full py-1 px-2 shadow-md'>
                  <div className='w-6 h-6 rounded-full overflow-hidden bg-purple-600 flex items-center justify-center mr-2'>
                    <Image
                      src={hakiIcon}
                      alt='User profile'
                      width={24}
                      height={24}
                      className='object-cover'
                    />
                  </div>
                  <span className='text-xs text-gray-300'>@limrik</span>
                </div>
              </div>
            )}
          </div>

          {/* similarity score */}
          <div className='flex flex-col items-center p-4'>
            <h2 className='text-4xl font-bold text-green-400'>
              {uploadData.similarityScore.toFixed(2)}%
            </h2>
            <p className='text-sm text-gray-400'>similarity score</p>
          </div>

          {/* original post */}
          <div className='flex flex-row gap-2'>
            {uploadData.editedImage && (
              <div className='w-1/2 relative overflow-hidden'>
                <div className='text-md'>Edited Image</div>
                <div className='aspect-w-16 aspect-h-9'>
                  <img
                    src={`data:image/png;base64,${uploadData.editedImage}`}
                    alt='Edited Image'
                    style={{ width: '400px', height: '225px' }}
                  />
                </div>
              </div>
            )}
            {uploadData.parentImage && (
              <div className='w-1/2 relative'>
                <div className='text-md'>Parent Content</div>
                <div className='aspect-w-16 aspect-h-9 '>
                  {/* <Image
                        src={URL.createObjectURL(uploadData.file)}
                        alt='Uploaded preview'
                        width={400}
                        height={225}
                        className='w-full h-full object-cover rounded-lg'
                      /> */}
                  <img
                    src={`data:image/png;base64,${uploadData.parentImage}`}
                    alt='Parent Image'
                    style={{ width: '400px', height: '225px' }}
                  />
                  <div className='absolute translate-y-1/4 bottom-0 right-2 flex items-center bg-gray-700 rounded-full py-1 px-2 shadow-md'>
                    <div className='w-6 h-6 rounded-full overflow-hidden bg-purple-600 flex items-center justify-center mr-2'>
                      <Image
                        src={uploadData.userIcon}
                        alt='User profile'
                        width={24}
                        height={24}
                        className='object-cover'
                      />
                    </div>
                    <span className='text-xs text-gray-300'>
                      @{uploadData.userHandle}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
          <form onSubmit={handleDisputeSubmit} className='mt-10'>
            <div className='mb-1'>
              <label
                htmlFor='disputeReason'
                className='block text-sm font-medium text-gray-400 mb-2'
              >
                Reason for Dispute:
              </label>
              <textarea
                id='disputeReason'
                value={disputeReason}
                onChange={(e) => setDisputeReason(e.target.value)}
                placeholder='Reason for Dispute'
                className='w-full text-black h-24 p-2 border border-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-600 resize-none'
              />
            </div>
            <button
              type='submit'
              className='w-full bg-purple-600 text-white font-semibold py-2 rounded-lg hover:bg-purple-700 transition duration-300'
            >
              Submit
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
