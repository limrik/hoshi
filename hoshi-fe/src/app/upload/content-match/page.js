"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { ChevronLeft } from "lucide-react";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";
import hakiIcon from "../../media/user-icon/haki.png";
import { useUpload } from "@/app/providers/UploadProvider";
import { Canvas, useThree } from "@react-three/fiber";
import { Stars } from "@/app/components/derivative-tree-components/Stars";
import { Graph } from "@/app/components/derivative-tree-components/Graph";
import { OrbitControls } from "@react-three/drei";
import { PinataSDK } from "pinata-web3";
import {
  HOSHINFT_ABI,
  HOSHINFT_CONTRACT_ADDRESS,
} from "../../../../contracts/hoshiNFT/hoshiNFT";
import { useDynamicContext } from "@dynamic-labs/sdk-react-core";
import { sepolia } from "viem/chains";
import { readContract, writeContract } from "viem/actions";
import Posts from "../../../../public/db/posts.json";
import Users from "../../../../public/db/users.json";
import { iliad } from "@/app/page";

export default function ContentMatchPage() {
  const router = useRouter();
  const [stage, setStage] = useState("initial");
  const [isDerivative, setIsDerivative] = useState(false);
  const { uploadData, setUploadData } = useUpload();
  const [editedImage, setEditedImage] = useState("");
  const [parentImage, setParentImage] = useState("");
  const [similarityScore, setSimilarityScore] = useState();
  const { primaryWallet } = useDynamicContext();
  const [userHandle, setUserHandle] = useState("");
  const [userIcon, setUserIcon] = useState("");
  const [parentTokenId, setParenTokenId] = useState(null);

  const pinata = new PinataSDK({
    pinataJwt: process.env.NEXT_PUBLIC_PINATA_JWT,
    pinataGateway: process.env.NEXT_PUBLIC_PINATA_GATEWAY,
  });

  const createNode = (
    id,
    size,
    text,
    date,
    authorName,
    authorIcon,
    media = null,
    children = []
  ) => ({
    id,
    size,
    text,
    date,
    authorName,
    authorIcon,
    media,
    children,
  });

  const data = createNode(
    "root",
    0.5,
    "This is the first post but I think it is a good one. Somehow or rather ",
    "2023-06-01",
    "John Doe",
    hakiIcon,
    hakiIcon,
    []
  );

  const CameraController = () => {
    const { camera, gl } = useThree();
    const controlsRef = useRef();

    useEffect(() => {
      if (controlsRef.current) {
        // Assuming your node is at position [0, 0, 0]
        controlsRef.current.target.set(0, 0, 0);
        controlsRef.current.update();
      }
    }, []);

    return (
      <OrbitControls
        ref={controlsRef}
        args={[camera, gl.domElement]}
        enablePan={false}
        enableZoom={true}
        enableRotate={true}
        autoRotate={true} // Enable auto-rotation
        autoRotateSpeed={1} // Adjust the speed as needed
        minDistance={7}
        maxDistance={20}
      />
    );
  };

  async function searchAPI(text, file, component) {
    // return {
    //   // file: 'data/images/image_20.png',
    //   file: '../db/media/gojo.png',
    //   parent_type: 'media',
    //   score: 0.6549215912818909,
    //   edited_media:
    //     'iVBORw0KGgoAAAANSUhEUgAAAs0AAAOYCAIAAAChGVkmAAEAAE…6yafJYw36WG9bBKwIAPwv2snLcd0N/ZkAAAAASUVORK5CYII=',
    //   parent_img:
    //     'iVBORw0KGgoAAAANSUhEUgAAA/kAAAdZCAYAAACgFtrxAAEAAE…8Y2Njg1dffbWbhroL5/8PANTN0wzDeskAAAAASUVORK5CYII=',
    //   token_id: 5,
    // };

    // return {
    const formData = new FormData();

    formData.append("text", text);
    formData.append("component", component);
    formData.append("file", file);

    try {
      const response = await fetch(
        "https://e3c4-104-244-25-79.ngrok-free.app/search",
        {
          method: "POST",
          body: formData,
          headers: {
            "ngrok-skip-browser-warning": "69420",
          },
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error Response:", errorData);
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Success:", data);
      return data;
    } catch (error) {
      console.error("Error:", error);
    }
  }

  useEffect(() => {
    async function performSearch() {
      setStage("checking");
      try {
        const res = await searchAPI(
          uploadData.content,
          uploadData.file,
          uploadData.component
        );

        // get back parent and similarity score
        setEditedImage(res.edited_media);
        setParentImage(res.parent_img);
        setParenTokenId(res.token_id);

        const post = Posts.find((post) => post.token_id === res.token_id);
        console.log(post);
        const user = Users.find(
          (user) => user.hoshiHandle === post.user_handle
        );
        // Now you have the correct user based on the post's user_handle
        setUserHandle(user.hoshiHandle);

        setUserIcon(`/media/${user.imagePath}`);
        setUploadData((prevData) => ({
          ...prevData,
          parentImage: res.parent_img,
          editedImage: res.edited_media,
          userHandle: Posts[res.token_id].user_handle,
          userIcon: `/media/${Users[Posts[res.token_id].user_handle]}`,
          similarityScore: parseFloat((res.score * 100).toFixed(2)),
        }));
        if (res.score > 0.2) {
          setIsDerivative(true);
        }
        setSimilarityScore(parseFloat((res.score * 100).toFixed(2)));
        setStage("done");
      } catch (error) {
        console.error("Error during searchAPI:", error);
        setStage("error");
      }
    }

    performSearch();
  }, [uploadData.content, uploadData.file, uploadData.component]);

  async function addDataToIPFS() {
    try {
      const { IpfsHash, PinSize, Timestamp } = await pinata.upload.file(
        uploadData.file
      );
      console.log(`IPFS: ${IpfsHash}`);
      const IPFSTokenURI = `https://gateway.pinata.cloud/ipfs/${IpfsHash}`;
      return IPFSTokenURI;
    } catch (error) {
      console.error("Error adding data to IPFS:", error);
    }
  }

  async function getParentTokenIds(walletClient, parentTokenID) {
    try {
      const parents = await readContract(walletClient, {
        address: HOSHINFT_CONTRACT_ADDRESS,
        abi: HOSHINFT_ABI,
        functionName: "getParents",
        args: [parentTokenID],
        chain: iliad,
      });
      console.log("parents: ", parents);
      return parents;
    } catch (error) {
      console.error("Error interacting with the contract:", error);
    }
  }

  async function getSimilarityScores(walletClient, parentTokenID) {
    try {
      const similarityScores = await readContract(walletClient, {
        address: HOSHINFT_CONTRACT_ADDRESS,
        abi: HOSHINFT_ABI,
        functionName: "getSimilarityScores",
        args: [parentTokenID],
        chain: iliad,
      });
      console.log("similarity scores: ", similarityScores);
      return similarityScores;
    } catch (error) {
      console.error("Error interacting with the contract:", error);
    }
  }

  async function mintNFT(
    walletClient,
    parentTokenIds,
    similarityScoresInput,
    IPFSTokenURI
  ) {
    try {
      const [address] = await walletClient.getAddresses();

      const tx = await writeContract(walletClient, {
        address: HOSHINFT_CONTRACT_ADDRESS,
        abi: HOSHINFT_ABI,
        functionName: "mintNFT",
        args: [address, parentTokenIds, similarityScoresInput, IPFSTokenURI],
        chain: iliad,
      });
      console.log("Subscription transaction sent:", tx);

      const tokenId = await readContract(walletClient, {
        address: HOSHINFT_CONTRACT_ADDRESS,
        abi: HOSHINFT_ABI,
        functionName: "nextTokenId",
        chain: iliad,
      });
      console.log("Token ID:", tokenId);

      // now i want to register the newly created nft on IPRegistery

      // if it is an original work, register it as a non commercial IP with the register workflow

      // if it is a derivative, get the parent

      return tokenId;
    } catch (error) {
      console.error("Error interacting with the contract:", error);
    }
  }

  // upload: text, file, token_id, user_id

  async function uploadAPI(text, file, token_id, user_id) {
    const formData = new FormData();

    formData.append("text", text);
    formData.append("file", file);
    formData.append("token_id", token_id);
    formData.append("user_id", user_id);
    try {
      const response = await fetch(
        "https://e3c4-104-244-25-79.ngrok-free.app/upload",
        {
          method: "POST",
          body: formData,
          headers: {
            "ngrok-skip-browser-warning": "69420",
          },
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error Response:", errorData);
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Success:", data);
      return data;
    } catch (error) {
      console.error("Error:", error);
    }
  }

  async function getPosts() {
    console.log("POSTS");
    try {
      const response = await fetch(
        "https://e3c4-104-244-25-79.ngrok-free.app/posts",
        {
          method: "GET",
          headers: {
            "ngrok-skip-browser-warning": "69420",
          },
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error Response:", errorData);
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Success:", data);
      return data;
    } catch (error) {
      console.error("Error:", error);
      return error;
    }
  }

  const handlePost = async () => {
    console.log("isDerivative:" + isDerivative);

    let parents = [];
    let similarityScores = [];

    // const parentTokenID = 1;
    const walletClient = await primaryWallet.getWalletClient();
    // upload upload.file to ipfs via pinata
    const IPFSTokenURI = await addDataToIPFS();

    if (isDerivative) {
      parents = await getParentTokenIds(walletClient, parentTokenId);

      // 2. get similarityScoresInput
      similarityScores = await getSimilarityScores(walletClient, parentTokenId);

      parents.unshift(parentTokenId);
      similarityScores.unshift(Math.round(uploadData.similarityScore));
    }
    // mint NFT
    // 1. get parentTokenIds

    const tokenId = await mintNFT(
      walletClient,
      parents,
      similarityScores,
      IPFSTokenURI
    );
    console.log(tokenId);

    // const tokenId = 1;
    await uploadAPI(
      uploadData.content,
      uploadData.file,
      parseInt(tokenId, 10),
      userHandle
    );

    router.push("/");
  };

  return (
    <div className="fixed inset-x-0 top-0 bottom-16 flex flex-col bg-[#1C1C1E] text-white overflow-hidden">
      <header className="flex justify-between items-center p-4 border-b border-gray-700">
        <button
          className="text-purple-400"
          onClick={() => router.push("/upload")}
          aria-label="Go back"
        >
          <ChevronLeft size={24} />
        </button>
      </header>
      <main className="flex-1 flex flex-col overflow-y-auto m-10">
        <AnimatePresence mode="wait">
          {uploadData && (stage === "initial" || stage === "checking") && (
            <motion.div
              key="vertical"
              className="relative overflow-hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
            >
              {uploadData.file && (
                <div className="relative rounded-t-lg overflow-hidden">
                  <div className="aspect-w-16 aspect-h-9">
                    {uploadData.file.type.startsWith("video/") ? (
                      <video
                        src={URL.createObjectURL(uploadData.file)}
                        controls
                        className="object-cover h-auto mx-auto"
                      />
                    ) : (
                      <Image
                        src={URL.createObjectURL(uploadData.file)}
                        alt="Uploaded preview"
                        width={800}
                        height={450}
                        className="object-cover h-auto mx-auto"
                      />
                    )}
                  </div>
                </div>
              )}
              {uploadData.content && (
                <div>
                  <p className="bg-gray-800 p-4 rounded-b-lg whitespace-pre-wrap">
                    {uploadData.content}
                  </p>
                </div>
              )}

              {stage == "checking" && (
                <motion.div
                  className="absolute inset-x-0 bottom-0 bg-purple-500 opacity-20"
                  initial={{ height: "0%" }}
                  animate={{
                    height: ["0%", "100%", "0%"],
                  }}
                  transition={{
                    repeat: Infinity,
                    duration: 4,
                    ease: [0.45, 0, 0.55, 1],
                    times: [0, 0.5, 1],
                  }}
                />
              )}
            </motion.div>
          )}

          {/* user's post */}
          {uploadData && stage === "done" && isDerivative && (
            <motion.div
              key="horizontal"
              className="relative flex flex-col h-[60vh] justify-between"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              <div className="flex flex-row">
                {uploadData.file && (
                  <div className="w-1/2 relative overflow-hidden rounded-l-lg">
                    <div className="aspect-w-16 aspect-h-9">
                      {uploadData.file.type.startsWith("video/") ? (
                        <video
                          src={URL.createObjectURL(uploadData.file)}
                          controls
                          className="object-cover h-auto mx-auto"
                        />
                      ) : (
                        <Image
                          src={URL.createObjectURL(uploadData.file)}
                          alt="Uploaded preview"
                          width={400}
                          height={225}
                          className="object-cover h-auto mx-auto"
                        />
                      )}
                    </div>
                  </div>
                )}
                {uploadData.content && (
                  <div className="w-1/2 p-4 flex bg-gray-800 rounded-r-lg relative text-xs">
                    <p className="whitespace-pre-wrap">{uploadData.content}</p>
                    <div className="absolute translate-y-1/4 bottom-0 right-2 flex items-center bg-gray-700 rounded-full py-1 px-2 shadow-md">
                      <div className="w-6 h-6 rounded-full overflow-hidden bg-purple-600 flex items-center justify-center mr-2">
                        <Image
                          src={hakiIcon}
                          alt="User profile"
                          width={24}
                          height={24}
                          className="object-cover"
                        />
                      </div>
                      <span className="text-xs text-gray-300">@limrik</span>
                    </div>
                  </div>
                )}
              </div>

              {/* similarity score */}
              <div className="flex flex-col items-center p-4">
                <h2 className="text-4xl font-bold text-green-400">
                  {similarityScore.toFixed(2)}%
                </h2>
                <p className="text-sm text-gray-400">similar</p>
              </div>

              {/* original post */}
              <div className="flex flex-row gap-2">
                {editedImage && (
                  <div className="w-1/2 relative overflow-hidden">
                    <div className="text-sm text-center mb-1">
                      Edited Content
                    </div>
                    <div className="aspect-w-16 aspect-h-9">
                      {/* <Image
                        src={URL.createObjectURL(uploadData.file)}
                        alt='Uploaded preview'
                        width={400}
                        height={225}
                        className='w-full h-full object-cover rounded-lg'
                      /> */}
                      {uploadData.file.type.startsWith("video/") ? (
                        <video
                          src={`data:video/mp4;base64,${editedImage}`}
                          controls
                          className="object-cover h-auto mx-auto"
                        />
                      ) : (
                        <img
                          src={`data:image/png;base64,${editedImage}`}
                          alt="Edited Image"
                          style={{ width: "400px", height: "225px" }}
                        />
                      )}
                      {/* <img
                        src={`data:image/png;base64,${editedImage}`}
                        alt='Edited Image'
                        style={{ width: '400px', height: '225px' }}
                      /> */}
                    </div>
                  </div>
                )}
                {parentImage && (
                  <div className="w-1/2 relative">
                    <div className="text-sm text-center mb-1">
                      Parent Content
                    </div>
                    <div className="aspect-w-16 aspect-h-9 ">
                      {/* <Image
                        src={URL.createObjectURL(uploadData.file)}
                        alt='Uploaded preview'
                        width={400}
                        height={225}
                        className='w-full h-full object-cover rounded-lg'
                      /> */}
                      {/* {uploadData.file.type.startsWith('video/') ? (
                        <video
                          src={`data:image/png;base64,${parentImage}`}
                          controls
                          className='object-cover h-auto mx-auto'
                        />
                      ) : ( */}
                      <img
                        src={`data:image/png;base64,${parentImage}`}
                        alt="Edited Image"
                        style={{ width: "400px", height: "225px" }}
                      />
                      <div className="absolute translate-y-1/4 bottom-0 right-2 flex items-center bg-gray-700 rounded-full py-1 px-2 shadow-md">
                        <div className="w-6 h-6 rounded-full overflow-hidden bg-purple-600 flex items-center justify-center mr-2">
                          <Image
                            src={userIcon}
                            alt="User profile"
                            width={24}
                            height={24}
                            className="object-cover"
                          />
                        </div>
                        <span className="text-xs text-gray-300">
                          @{userHandle}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
                {/* {uploadData.content && (
                  <div className='w-1/2 p-4 flex bg-gray-800 rounded-r-lg relative'>
                    <p className='whitespace-pre-wrap'>{uploadData.content}</p>
                   
                  </div>
                )} */}
              </div>
              <div className="flex flex-col gap-2 mt-10">
                <div className="flex gap-2">
                  <button
                    className="bg-purple-400 hover:bg-purple-500 text-white px-4 py-4 rounded-lg font-semibold text-sm flex-1 overflow-hidden transition-transform duration-300"
                    onClick={() => router.push("/derivative-tree")}
                  >
                    <span className="block transform hover:scale-105">
                      Derivative Tree
                    </span>
                  </button>

                  <button
                    className="bg-purple-400 hover:bg-purple-500 text-white px-4 py-4 rounded-lg font-semibold text-sm flex-1 overflow-hidden transition-transform duration-300"
                    onClick={() => router.push("/dispute")}
                  >
                    <span className="block transform hover:scale-105">
                      Submit a Dispute
                    </span>
                  </button>
                </div>
                <button
                  className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-4 rounded-lg font-semibold text-sm mt-4 overflow-hidden transition-transform duration-300"
                  onClick={handlePost}
                >
                  <span className="block transform hover:scale-105">Post</span>
                </button>
              </div>
            </motion.div>
          )}

          {uploadData && stage === "done" && !isDerivative && (
            <div className="flex flex-col gap-10 h-full justify-between">
              <h2>
                Congratulations! Your post has been deemed as original and added
                to the hoshi ecosystem!
              </h2>

              <motion.div
                key="horizontal"
                className="relative flex flex-col h-[30vh] justify-between"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <Canvas
                  style={{ background: "#000" }}
                  camera={{ position: [0, 0, 15], fov: 60 }}
                >
                  <ambientLight intensity={0.8} />
                  <pointLight position={[10, 10, 10]} intensity={0.5} />
                  <Stars />
                  <Graph data={data} />
                  <CameraController />
                </Canvas>
              </motion.div>

              <motion.div
                key="horizontal"
                className="relative flex flex-col h-[25vh] justify-between"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <div className="flex flex-row">
                  {uploadData.file && (
                    <div className="w-1/2 relative overflow-hidden rounded-l-lg">
                      <div className="aspect-w-16 aspect-h-9">
                        {uploadData.file.type.startsWith("video/") ? (
                          <video
                            src={URL.createObjectURL(uploadData.file)}
                            controls
                            className="object-cover h-auto mx-auto"
                          />
                        ) : (
                          <Image
                            src={URL.createObjectURL(uploadData.file)}
                            alt="Uploaded preview"
                            width={400}
                            height={225}
                            className="object-cover h-auto mx-auto"
                          />
                        )}
                      </div>
                    </div>
                  )}
                  {uploadData.content && (
                    <div className="w-1/2 p-4 flex bg-gray-800 rounded-r-lg relative">
                      <p className="whitespace-pre-wrap">
                        {uploadData.content}
                      </p>
                      <div className="absolute translate-y-1/4 bottom-0 right-2 flex items-center bg-gray-700 rounded-full py-1 px-2 shadow-md">
                        <div className="w-6 h-6 rounded-full overflow-hidden bg-purple-600 flex items-center justify-center mr-2">
                          <Image
                            src={hakiIcon}
                            alt="User profile"
                            width={24}
                            height={24}
                            className="object-cover"
                          />
                        </div>
                        <span className="text-xs text-gray-300">@user</span>
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>

              <button
                className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-4 rounded-lg font-semibold text-sm transition transform hover:scale-105 duration-300 mt-4"
                onClick={handlePost}
              >
                Return
              </button>
            </div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
