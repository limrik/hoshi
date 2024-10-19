'use client';

import { createContext, useContext, useState } from 'react';

// Create the context
const UploadContext = createContext();

// Export the hook for easy access
export const useUpload = () => {
  const context = useContext(UploadContext);
  if (!context) {
    throw new Error('useUpload must be used within an UploadProvider');
  }
  return context;
};

// Create and export the provider component
export const UploadProvider = ({ children }) => {
  const [uploadData, setUploadData] = useState({
    content: '',
    file: null,
    component: '',
  });

  return (
    <UploadContext.Provider value={{ uploadData, setUploadData }}>
      {children}
    </UploadContext.Provider>
  );
};
