'use client';
import { useRef } from 'react';
import { useVideo } from './providers/VideoProvider';

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
    </div>
  );
}
