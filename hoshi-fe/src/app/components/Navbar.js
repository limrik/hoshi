'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Home, Search, Upload, User } from 'lucide-react';

export default function Navbar() {
  const pathname = usePathname();

  const NavItem = ({ href, icon: Icon }) => {
    const isActive =
      pathname === href || (href !== '/' && pathname.startsWith(href));
    return (
      <Link
        href={href}
        className={`flex flex-col items-center ${
          isActive ? 'text-purple-400' : 'text-gray-500'
        }`}
      >
        <Icon size={24} />
      </Link>
    );
  };

  return (
    <nav className='fixed bottom-0 left-0 right-0 flex justify-around items-center h-16 bg-[#1C1C1E] border-t border-gray-700 z-40'>
      <NavItem href='/' icon={Home} />
      {/* <NavItem href='/search' icon={Search} /> */}
      <NavItem href='/upload' icon={Upload} />
      <NavItem href='/profile' icon={User} />
    </nav>
  );
}
