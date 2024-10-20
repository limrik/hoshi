"use client";

import { ChevronLeft, LockIcon, WalletIcon, UnlockIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import { X, Check } from "lucide-react";
import { motion } from "framer-motion";
import confetti from "canvas-confetti";
import { useDynamicContext } from "@dynamic-labs/sdk-react-core";
import { sepolia } from "viem/chains";
import { readContract, writeContract } from "viem/actions";
import { formatUnits } from "viem";
import {
  HOSHITOKEN_CONTRACT_ADDRESS,
  HOSHITOKEN_ABI,
} from "../../../../contracts/hoshitoken/hoshitoken";
import Drawer from "@/app/components/Drawer";
import { iliad } from "@/app/page";

export default function CreatorEarningsDashboard() {
  const router = useRouter();
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isUnwrapped, setIsUnwrapped] = useState(false);
  const [earningsValue, setEarningsValue] = useState();
  const [showConfetti, setShowConfetti] = useState(false);
  const { primaryWallet } = useDynamicContext();
  const [isSubscribed, setIsSubscribed] = useState(false);
  const [walletClient, setWalletClient] = useState();
  const [address, setAddress] = useState();

  useEffect(() => {
    if (showConfetti) {
      const timer = setTimeout(() => setShowConfetti(false), 5000);
      return () => clearTimeout(timer);
    }
  }, [showConfetti]);

  const fireConfetti = () => {
    confetti({
      particleCount: 150,
      spread: 70,
      origin: { y: 0.6 },
    });
  };

  const handleUnwrap = () => {
    const unwrapHoshitokens = async () => {
      try {
        const tx = await writeContract(walletClient, {
          address: HOSHITOKEN_CONTRACT_ADDRESS,
          abi: HOSHITOKEN_ABI,
          functionName: "unwrapTokens",
          args: [100000 * 10 ** 18],
          chain: iliad,
        });

        console.log("Subscription transaction sent:", tx);

        const receipt = await tx.wait();
        console.log("Transaction confirmed:", receipt);

        alert("Subscription successful!");
      } catch (error) {
        console.error("Error interacting with the contract:", error);
      }
    };
    unwrapHoshitokens();
    setIsUnwrapped(true);
    setEarningsValue(0);
    fireConfetti();
    setIsDrawerOpen(false);
  };

  useEffect(() => {
    const getHoshitokens = async () => {
      try {
        const walletClient = await primaryWallet.getWalletClient();
        const [address] = await walletClient.getAddresses();

        setWalletClient(walletClient);
        setAddress(address);

        const hoshitokens = await readContract(walletClient, {
          address: HOSHITOKEN_CONTRACT_ADDRESS,
          abi: HOSHITOKEN_ABI,
          functionName: "balanceOf",
          args: [address],
          chain: iliad,
        });

        setEarningsValue(formatUnits(hoshitokens, 18));

        const subscribedCreator = await readContract(walletClient, {
          address: HOSHITOKEN_CONTRACT_ADDRESS,
          abi: HOSHITOKEN_ABI,
          functionName: "subscribedCreators",
          args: [address],
          chain: iliad,
        });

        setIsSubscribed(subscribedCreator);
      } catch (error) {
        console.error("Error interacting with the contract:", error);
      }
    };

    getHoshitokens();
  }, [primaryWallet]);

  const handleSubscribe = async () => {
    try {
      const subscriptionFee = await readContract(walletClient, {
        address: HOSHITOKEN_CONTRACT_ADDRESS,
        abi: HOSHITOKEN_ABI,
        functionName: "subscriptionFeeETH",
        chain: iliad,
      });

      console.log("Subscription Fee (in Wei):", subscriptionFee.toString());

      const tx = await writeContract(walletClient, {
        address: HOSHITOKEN_CONTRACT_ADDRESS,
        abi: HOSHITOKEN_ABI,
        functionName: "subscribeCreator",
        chain: iliad,
        account: address,
        value: subscriptionFee,
      });

      console.log("Subscription transaction sent:", tx);

      const receipt = await tx.wait();
      console.log("Transaction confirmed:", receipt);

      alert("Subscription successful!");
    } catch (error) {
      console.error("Error interacting with the contract:", error);
    }

    setIsDrawerOpen(false);
  };

  return (
    <div className="fixed inset-x-0 top-0 bottom-16 z-50 flex flex-col bg-[#1C1C1E] text-white">
      <header className="flex justify-between items-center p-4 border-b border-gray-700">
        <button
          className="text-purple-400"
          onClick={() => router.push("/profile")}
          aria-label="Go back"
        >
          <ChevronLeft size={24} />
        </button>
      </header>
      <main>
        <div className="flex flex-col gap-4">
          <p className="text-sm text-gray-400 p-8">
            As a creator, subscribing gives you access to the full power of your
            content. By subscribing, you unlock the ability to cash out your IP
            and start earning ad revenue based on the engagement your posts
            generate.
          </p>
          <div
            className={`mx-16 bg-gradient-to-br from-gray- 00 to-gray-800 text-white shadow-xl rounded-lg overflow-hidden ${
              isSubscribed
                ? "bg-gradient-to-br from-purple-600 to-purple-900"
                : "bg-gradient-to-br from-gray-900 to-gray-800"
            }`}
          >
            <div className="p-6">
              <div
                className={`flex justify-between items-center mb-4 ${
                  isSubscribed ? "text-white" : "text-gray-400"
                }`}
              >
                <span className="text-sm font-medium">Oct 2024</span>
                {isSubscribed ? (
                  <UnlockIcon className="h-5 w-5 text-yellow-500" />
                ) : (
                  <LockIcon className="h-5 w-5 text-yellow-500" />
                )}
              </div>
              <div className="flex items-baseline mb-2">
                <motion.span
                  className="text-3xl font-bold"
                  animate={{ opacity: isUnwrapped ? 0.5 : 1 }}
                >
                  <motion.span
                    animate={{ number: isUnwrapped ? 0 : earningsValue }}
                    transition={{ duration: 1, ease: "easeInOut" }}
                  >
                    {earningsValue}
                  </motion.span>
                </motion.span>
                <span
                  className={`ml-2 text-lg ${
                    isSubscribed ? "text-white" : "text-gray-400"
                  }`}
                >
                  hoshitokens
                </span>
              </div>
            </div>
            <div className="flex flex-col space-y-3 p-6 pt-0">
              <button
                onClick={() => setIsDrawerOpen(true)}
                className={`w-full ${
                  isSubscribed
                    ? "bg-yellow-500 hover:bg-yellow-600 text-purple-900 font-bold"
                    : "bg-purple-600 hover:bg-purple-700 text-white"
                } py-2 px-4 rounded-md transition-colors duration-200 mb-3`}
              >
                {isSubscribed ? "Cash out" : "Subscribe to Cash out"}
              </button>
              <Drawer
                isOpen={isDrawerOpen}
                onClose={() => setIsDrawerOpen(false)}
                className="z-60"
              >
                {isSubscribed ? (
                  <div className="flex flex-col h-full">
                    <div className="flex justify-between items-center mb-6">
                      <h2 className="text-4xl font-bold">hoshi</h2>
                      <button
                        onClick={() => setIsDrawerOpen(false)}
                        className="text-white hover:bg-purple-800 p-2 rounded-full"
                      >
                        <X className="h-6 w-6" />
                      </button>
                    </div>
                    <p className="text-sm mb-6">
                      To complete your cash-out, please confirm your wallet
                      address. Once confirmed, you&apos;ll be able to unwrap
                      your tokens and withdraw your earnings.
                    </p>
                    <h3 className="text-lg font-semibold mb-2">
                      Confirm Your Wallet Address
                    </h3>
                    <div className="flex items-center bg-gray-800 rounded-lg p-3 mb-2">
                      <span className="text-sm mr-2 truncate flex-grow">
                        {primaryWallet.address}
                      </span>
                    </div>
                    <div className="bg-purple-800 rounded-lg p-4 mb-6">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm">Oct 2024</span>
                        <span className="text-2xl font-bold">
                          ${earningsValue} hoshitokens
                        </span>
                      </div>
                      <span className="text-sm text-purple-300">
                        Ad Revenue
                      </span>
                    </div>
                    <button
                      className="bg-yellow-500 hover:bg-yellow-600 text-purple-900 font-bold py-3 px-4 rounded-lg transition-colors duration-200"
                      onClick={handleUnwrap}
                    >
                      Unwrap Tokens
                    </button>
                  </div>
                ) : isUnwrapped ? (
                  <div className="flex flex-col h-full">
                    <div className="flex justify-between items-center mb-6">
                      <h2 className="text-4xl font-bold">hoshi</h2>
                      <button
                        onClick={() => setIsDrawerOpen(false)}
                        className="text-white hover:bg-purple-800 p-2 rounded-full"
                      >
                        <X className="h-6 w-6" />
                      </button>
                    </div>
                    <div className="flex-grow flex flex-col justify-center items-center text-center">
                      <p className="text-xl font-semibold mb-2">
                        🎉 Congratulations!
                      </p>
                      <p className="text-sm mb-4">
                        You have successfully unwrapped your tokens and cashed
                        out {earningsValue} hoshitokens to your wallet
                      </p>
                    </div>
                    <button
                      className="w-full bg-yellow-500 text-purple-900 font-bold py-3 px-4 rounded-lg mt-auto"
                      onClick={() => {
                        router.push("/profile");
                      }}
                    >
                      Back to Profile
                    </button>
                  </div>
                ) : (
                  <div>
                    <div className="flex justify-between items-center mb-6">
                      <h2 className="text-4xl font-bold">hoshi</h2>
                      <button
                        onClick={() => setIsDrawerOpen(false)}
                        className="text-white hover:bg-purple-800 p-2 rounded-full"
                      >
                        <X className="h-6 w-6" />
                      </button>
                    </div>
                    <h3 className="text-xl font-semibold mb-4">
                      Unlock Ad Revenue Sharing
                    </h3>
                    <ul className="space-y-2 mb-6">
                      {[
                        "Cash Out Your IP",
                        "Get paid monthly",
                        "Access premium analytics tools",
                      ].map((item, index) => (
                        <li key={index} className="flex items-center">
                          <Check className="h-5 w-5 mr-2 text-yellow-400" />
                          {item}
                        </li>
                      ))}
                    </ul>
                    <div className="flex space-x-4 mb-6">
                      {[
                        {
                          title: "Monthly",
                          price: "$20.00",
                          desc: "Billed Monthly",
                        },
                        {
                          title: "Yearly",
                          price: "$200.00",
                          desc: "Free 1 Week Trial",
                          extra: "Save $40.00",
                        },
                      ].map((plan, index) => (
                        <div
                          key={index}
                          className={`flex-1 p-4 bg-purple-800 rounded-lg ${
                            index === 0 ? "border-2 border-yellow-400" : ""
                          }`}
                        >
                          <h4 className="font-semibold mb-2">{plan.title}</h4>
                          <p className="text-2xl font-bold mb-1">
                            {plan.price}
                          </p>
                          <p className="text-sm">{plan.desc}</p>
                          {plan.extra && (
                            <p className="text-sm text-yellow-400">
                              {plan.extra}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                    <button
                      className="w-full bg-yellow-500 hover:bg-yellow-600 text-purple-900 font-semibold py-3 rounded-lg transition-colors duration-200"
                      onClick={handleSubscribe}
                    >
                      Subscribe
                    </button>
                  </div>
                )}
              </Drawer>
              <button
                className={`w-full py-2 px-4 rounded-md transition-colors duration-200 flex items-center justify-center ${
                  isSubscribed
                    ? "text-yellow-400 hover:text-yellow-300 hover:bg-purple-700"
                    : "text-purple-400 hover:text-purple-300 hover:bg-gray-700"
                }`}
              >
                <WalletIcon className="mr-2 h-4 w-4" />
                Manage Wallet
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
