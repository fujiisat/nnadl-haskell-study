{-# LANGUAGE BangPatterns #-}

{- |
Module      : Main
Description : Haskell translation of Neural Networks and Deep Learning code sample
Copyright   : (c) FUJII Satohi, 2016
License     : BSD3
Maintainer  : fujiisat@gmaill.com
Stability   : experimental
Portability : portable

My personal Haskell translation of Neural Networks and Deep Learning
code sample
(https://github.com/mnielsen/neural-networks-and-deep-learning,
Copyright (c) 2012-2015 Michael Nielsen)

-}

module Main where

import           Control.Monad         (replicateM, zipWithM)
import           Data.List             (foldl')
import qualified Data.Vector           as V
import           Numeric.LinearAlgebra (( #> ))
import qualified Numeric.LinearAlgebra as H
import           System.Environment    (getArgs)
import           System.Random         (newStdGen, randomIO)
import           System.Random.Shuffle (shuffle')
import           Text.Printf           (printf)

import           MnistData

type Vec = H.Vector H.R
type Mat = H.Matrix H.R
type Data = (Vec, Vec)
data Network = Network
               { numLayers :: !Int
               , sizes     :: V.Vector Int
               , biases    :: V.Vector Vec
               , weights   :: V.Vector Mat
               } deriving (Eq, Show)

main :: IO ()
main = do
  -- ([30, 10, 3.0], [784, 30, 10])
  ([epo, mbsize, e], networkArgs) <-  splitAt 3  <$> getArgs
  putStrLn $ "Epoch: " ++ epo
  putStrLn $ "Minibatch: " ++ mbsize
  putStrLn $ "Eta: " ++ e
  putStrLn $ "Network: " ++ show networkArgs
  (trainingData, validateDate, testData) <- loadDataWrapper
  network <- networkInit (map read networkArgs)
  !result <- sgd network (V.fromList trainingData) (read epo) (read mbsize) (read e) (Just $ V.fromList testData)
  -- print result
  return ()

networkInit :: [Int] -> IO Network
networkInit sizes = do
  b <- map H.flatten <$> mapM (`H.randn` 1) (tail sizes)
  w <- zipWithM H.randn sizes (tail sizes)
  return Network { numLayers = length sizes
                 , sizes     = V.fromList sizes
                 , biases    = V.fromList b
                 , weights   = V.fromList w
                 }

feedforward :: Network -> Vec -> Vec
feedforward (Network n s biases weights) a =
  V.foldl' (\ !a (w, b) -> sigmoidVec ((H.tr w #> a) + b) ) a $ V.zip weights biases

sgd :: Network
    -> V.Vector Data               -- Training data
    -> Int                         -- Epoch
    -> Int                         -- Minibatch size
    -> Double                      -- Eta
    -> Maybe (V.Vector (Vec, Int)) -- Test data
    -> IO Network
sgd network trainingData 0 miniBatchSize eta testData = return network
sgd network trainingData epohcs miniBatchSize eta testData = do
  miniBatches <- divide miniBatchSize <$> shuffleList trainingData
  let network' = V.foldl' (\ !n b -> updateMiniBatch n b eta) network miniBatches
  case testData of
    Just t  -> putStrLn $ printf "Epoch %d: %d / %d" epohcs (evaluate network' t) (length t)
    Nothing -> putStrLn $ printf "Epoch %d complete" epohcs
  sgd network' trainingData (epohcs - 1) miniBatchSize eta testData

updateMiniBatch :: Network -> V.Vector Data -> Double -> Network
updateMiniBatch network miniBatch eta = network { weights=weights', biases=biases'}
  where
    f :: (V.Vector Vec, V.Vector Mat) -> Data -> (V.Vector Vec, V.Vector Mat)
    f nabla mb = nablaNew
      where
        !deltaNabla = backprop network mb
        !nablaNew = ( V.zipWith (+) (fst nabla) (fst deltaNabla)
                    , V.zipWith (+) (snd nabla) (snd deltaNabla))

    (!nablaB, !nablaW) = V.foldl' f zeros miniBatch

    !zeros = ( V.map (\a -> a - a) $ biases network
             , V.map (\a -> a - a) $ weights network )

    !etaN = eta / fromIntegral (length miniBatch)
    !weights' = V.map (\(w, nw) -> w - H.cmap (etaN *) nw) $ V.zip (weights network) nablaW
    !biases'  = V.map (\(b, nb) -> b - H.cmap (etaN *) nb) $ V.zip (biases network) nablaB

backprop :: Network -> Data -> (V.Vector Vec, V.Vector Mat)
backprop network (x, y) = (V.fromList nablaB, V.fromList nablaW)
  where
    (ac0:ac1:activations, z:zs) = V.foldl' forward ([x], []) $ V.zip (biases network) (weights network)
--    !delta = costDerivative ac0 y * sigmoidPrimeVec z -- quadratic
    !delta = costDerivative ac0 y  -- cross entropy
    !nablaIni = ([delta], [H.tr (delta `H.outer` ac1)], delta)
    (nablaB, nablaW, _) = foldl' backward nablaIni
      $ zip3 zs (V.toList $ V.reverse $ weights network) activations

backward :: ([Vec], [Mat], Vec) -> (Vec, Mat, Vec) -> ([Vec], [Mat], Vec)
backward  (nabB, nabW, del) (z, w, ac) = (delta : nabB, H.tr (delta `H.outer` ac) : nabW, delta)
  where
    !sp = sigmoidPrimeVec z
    !delta = (w #> del) * sp

forward :: ([Vec], [Vec]) -> (Vec, Mat) -> ([Vec], [Vec])
forward (a@(ac:acs), zzs) (b, w) = (activation:a, z:zzs)
  where
    !z = (H.tr w #> ac) + b
    !activation = sigmoidVec z

costDerivative :: Vec -> Vec -> Vec
costDerivative outputActivations y = outputActivations - y

sigmoid :: Double -> Double
sigmoid z = 1.0 / (1.0 + exp (-z))

sigmoidVec :: Vec -> Vec
sigmoidVec = H.cmap sigmoid


sigmoidPrime :: Double -> Double
sigmoidPrime z = sigmoid z * (1 - sigmoid z)

sigmoidPrimeVec :: Vec -> Vec
sigmoidPrimeVec = H.cmap sigmoidPrime

evaluate :: Network -> V.Vector (Vec, Int) -> Int
evaluate network testData = length $ V.filter (uncurry (==)) testResults
  where
    testResults = V.map (\(x, y) -> (H.maxIndex (feedforward network x), y)) testData

divide :: Int -> V.Vector a -> V.Vector (V.Vector a)
divide n xs | V.null xs = V.empty
            | otherwise =  V.cons hd (divide n tl)
  where
    (hd, tl) = V.splitAt n xs

shuffleList :: V.Vector a -> IO (V.Vector a)
shuffleList xs = do
  let len = length xs
  gen <- newStdGen
  return $ V.fromList $ shuffle' (V.toList xs) len gen
