'use client';

import { createContext, useContext, useState } from 'react';

// Create the context
const VideoContext = createContext();

// Export the hook for easy access
export const useVideo = () => {
  const context = useContext(VideoContext);
  if (!context) {
    throw new Error('useUpload must be used within an UploadProvider');
  }
  return context;
};

// Create and export the provider component
export const VideoProvider = ({ children }) => {
  const [isFirstVisit, setIsFirstVisit] = useState(true);

  return (
    <VideoContext.Provider value={{ isFirstVisit, setIsFirstVisit }}>
      {children}
    </VideoContext.Provider>
  );
};
