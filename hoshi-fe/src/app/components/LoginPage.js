import { DynamicEmbeddedWidget } from '@dynamic-labs/sdk-react-core';
import Image from 'next/image';
import hoshiLogoLight from '../media/logo/hoshi-logo-light.png';

export default function LoginPage() {
  return (
    <div className='fixed w-full min-h-screen flex items-center justify-center p-4 gradient-background'>
      <style jsx global>{`
        @keyframes gradientShift {
          0% {
            background-position: 0% 50%;
          }
          50% {
            background-position: 100% 50%;
          }
          100% {
            background-position: 0% 50%;
          }
        }
        .gradient-background {
          background: radial-gradient(
              circle at 20% 20%,
              rgba(255, 250, 205, 0.4) 0%,
              rgba(255, 250, 205, 0) 50%
            ),
            radial-gradient(
              circle at 80% 80%,
              rgba(216, 191, 216, 0.4) 0%,
              rgba(216, 191, 216, 0) 50%
            ),
            radial-gradient(
              circle at 50% 50%,
              rgba(255, 161, 184, 0.4) 0%,
              rgba(255, 161, 184, 0) 50%
            ),
            linear-gradient(135deg, #6a5acd, #8b7beb, #fffacd);
          background-size: 200% 200%;
          animation: gradientShift 20s ease infinite;
        }
      `}</style>
      <div className='w-full max-w-md bg-zinc-100 bg-opacity-30 backdrop-blur-lg rounded-3xl shadow-lg overflow-hidden'>
        <div className='p-8 space-y-6'>
          <div className='flex items-center justify-center relative mx-auto'>
            <Image
              src={hoshiLogoLight}
              alt='Hoshi logo'
              width={100}
              height={140}
              unoptimized
            />
          </div>
          <DynamicEmbeddedWidget />
        </div>
      </div>
    </div>
  );
}
