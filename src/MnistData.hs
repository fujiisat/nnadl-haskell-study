{-# LANGUAGE BangPatterns #-}
module MnistData (loadDataWrapper) where

import           Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy   as BL
import qualified Numeric.LinearAlgebra  as H


loadDataWrapper :: IO ([(H.Vector H.R, H.Vector H.R)], [(H.Vector H.R, Int)], [(H.Vector H.R, Int)])
loadDataWrapper = do
  (_, _, testImage) <- loadImage "data/t10k-images-idx3-ubyte.gz"
  (_, testLabel) <- loadLabel "data/t10k-labels-idx1-ubyte.gz"
  (_, _, trainImage) <- loadImage "data/train-images-idx3-ubyte.gz"
  (_, trainLabel) <- loadLabel "data/train-labels-idx1-ubyte.gz"
  let !testData = zip testImage testLabel
  let (!trainData', !validateDate) = splitAt 50000 $ zip trainImage trainLabel
  let trainData = map (\(a, b)-> (a, labelToVec b)) trainData'
  return (trainData, validateDate, testData)

labelToVec :: Int -> H.Vector H.R
labelToVec n = H.fromList $ replicate n 0 ++ [1.0] ++ replicate (9-n) 0

loadLabel :: FilePath -> IO (Int, [Int])
loadLabel fp = do
  content <- decompress <$> BL.readFile fp
  let (hd, dat) = BL.splitAt 8 content
  let (magic, size) = BL.splitAt 4 hd
  case BL.unpack magic of
    [0, 0, 8, 1] -> return (readInt size, take (readInt size) (parseData dat))
    _            -> return (0, [])

loadImage :: FilePath -> IO (Int, (Int, Int), [H.Vector H.R])
loadImage fp = do
  content <- decompress <$> BL.readFile fp
  let (hd, dat) = BL.splitAt 16 content
  let (magic, tl) = BL.splitAt 4 hd
  let (size, xy) =  BL.splitAt 4 tl
  let (rowW, colW) = BL.splitAt 4 xy
  let (row, col) = (readInt rowW, readInt colW)
  let images = map (H.fromList . map ((/256.0). fromIntegral)) (every (row * col) $ BL.unpack dat)
  case BL.unpack magic of
    [0, 0, 8, 3] -> return (readInt size, (row, col), images)
    _            -> return (0, (0, 0), images)

parseData :: BL.ByteString -> [Int]
parseData = map fromIntegral . BL.unpack

readInt :: Integral a => BL.ByteString -> a
readInt bs = foldl (\a b -> a * 256 + b) 0 $ map fromIntegral $ BL.unpack bs

every :: Int -> [a] -> [[a]]
every n xs = hd : every n tl
  where
    (hd, tl) = splitAt n xs
