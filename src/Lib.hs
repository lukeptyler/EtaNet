{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
--{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
--{-# LANGUAGE AllowAmbiguousTypes #-}
--{-# LANGUAGE FunctionalDependencies #-}


module Lib where

import Data.Vector (Vector)
import qualified Data.Vector as V
import Data.Matrix

type family GradientData a
type instance GradientData (WrappedActivation x a) = ()
type instance GradientData (Dense a) = (Matrix a, Matrix a)

class InputRecord x where
  recordInput :: Matrix a -> x a -> x a
  getInput :: x a -> Matrix a

-- Layers
class (InputRecord x) => Layer x where
  forward :: Floating a => x a -> Matrix a -> (x a, Matrix a)
  backward :: Floating a => x a -> a -> Matrix a -> (x a, Matrix a)

  train :: Floating a => x a -> a -> GradientData (x a) -> x a
  train layer _ _ = layer



data Dense a = Dense {inputDense, weightsDense, biasDense :: Matrix a}
  deriving (Show)

instance InputRecord Dense where
  recordInput xs layer = layer {inputDense = xs}
  getInput = inputDense

instance Layer Dense where
  forward layer xs = (layer {inputDense = xs}, elementwise (+) (biasDense layer) $ multStd2 (weightsDense layer) xs)
  backward layer learningRate dEdY = (train layer learningRate gradData, multStd2 (transpose $ weightsDense layer) dEdY)
    where
      gradData = (multStd2 dEdY $ transpose $ inputDense layer, dEdY)

  train layer learningRate (weightsGrad, biasGrad) = layer 
    { weightsDense = elementwise (-) (weightsDense layer) (scaleMatrix learningRate weightsGrad)
    , biasDense = elementwise (-) (biasDense layer) (scaleMatrix learningRate biasGrad)
    }

-- Activations
class Activation x where
  activation :: Floating a => x a -> a -> a
  activationPrime :: Floating a => x a -> a -> a


newtype Tanh a = Tanh {inputTanh :: Matrix a}
  deriving (Show)

instance InputRecord Tanh where
  recordInput xs layer = layer {inputTanh = xs}
  getInput = inputTanh

instance Activation Tanh where
  activation _ x = tanh x
  activationPrime _ x = 1 - (tanh x)^2



-- Network
data Network a where
  N :: Network a
  L :: (Layer x, Floating a) => x a -> Network a -> Network a

instance Show (Network a) where
  show N = "END"
  show (L layer remainingLayers) = "Layer | " <> show remainingLayers

runNetwork :: Network a -> Matrix a -> Matrix a
runNetwork network = snd . runForward network

runForward :: Network a -> Matrix a -> (Network a, Matrix a)
runForward N            output = (N, output)
runForward (L layer ls) input  = (L layer' ls', output)
  where
    (layer', processed) = forward layer input
    (ls', output) = runForward ls processed

runBackward :: Network a -> a -> Matrix a -> (Network a, Matrix a)
runBackward N            _            dEdY = (N, dEdY)
runBackward (L layer ls) learningRate dEdY = (L layer' ls', output)
  where
    (ls', input) = runBackward ls learningRate dEdY
    (layer', output) = backward layer learningRate input

trainNetworkStep :: (Loss l, Floating a) => Network a -> l -> a -> Matrix a -> Matrix a -> Network a
trainNetworkStep network l learningRate input expected = fst $ runBackward toTrain learningRate dEdY
  where
    (toTrain, actual) = runForward network input
    dEdY = lossPrime l expected actual

trainNetworkEpoch :: (Loss l, Floating a) => Network a -> l -> a -> [Matrix a] -> [Matrix a] -> Network a
trainNetworkEpoch network l learningRate inputs expecteds = foldr (\(x,y) net -> trainNetworkStep net l learningRate x y) network $ zip inputs expecteds

trainNetwork :: (Loss l, Floating a) => Network a -> Int -> l -> a -> [Matrix a] -> [Matrix a] -> Network a
trainNetwork network epochs l learningRate inputs expecteds = foldr ($) network $ replicate epochs (\net -> trainNetworkEpoch net l learningRate inputs expecteds)

-- dense 2 3, tanh, dense 3 1, tanh
dense1 :: Dense Float
dense1 = Dense
  { inputDense = zero 2 1
  , weightsDense = fromList 3 2 [0.2,-1,0.75,0.4,-0.1,0]
  , biasDense = fromList 3 1 [0.5,-0.3,0.4]
  }

tanh_ :: WrappedActivation Tanh Float
tanh_ = WrappedActivation $ Tanh $ zero 1 1

dense2 :: Dense Float
dense2 = Dense
  { inputDense = zero 3 1
  , weightsDense = fromList 1 3 [-0.25,0.9,0.5]
  , biasDense = fromList 1 1 [1]
  }

inputs :: [Matrix Float]
inputs = 
  [ fromList 2 1 [0, 0]
  , fromList 2 1 [0, 1]
  , fromList 2 1 [1, 0]
  , fromList 2 1 [1, 1]
  ]

expecteds :: [Matrix Float]
expecteds = 
  [ fromList 1 1 [0]
  , fromList 1 1 [0]
  , fromList 1 1 [0]
  , fromList 1 1 [1]
  ]

net1 = L dense1 $ L tanh_ $ L dense2 $ L tanh_ N

runAll :: Network a -> [Matrix a] -> [Matrix a]
runAll network = map (runNetwork network)

-- Loss Functions
class Loss x where
  loss :: (Floating a) => x -> Matrix a -> Matrix a -> a
  lossPrime :: (Floating a) => x -> Matrix a -> Matrix a -> Matrix a

data MSE = MSE
instance Loss MSE where
  loss _ expected actual = sum $ elementwise (\x y -> (x - y)^2/fromIntegral (nrows expected)) expected actual
  lossPrime _ expected actual = mapPos (\_ x -> 2 * x / fromIntegral (nrows expected)) $ elementwise (-) actual expected
    
-- Wrappers
newtype WrappedActivation x a = WrappedActivation (x a)
  deriving (Show, Eq)

instance (InputRecord x) => InputRecord (WrappedActivation x) where
  recordInput xs (WrappedActivation layer) = WrappedActivation $ recordInput xs layer
  getInput (WrappedActivation layer) = getInput layer

instance (Activation x) => Activation (WrappedActivation x) where
  activation (WrappedActivation layer) = activation layer
  activationPrime (WrappedActivation layer) = activationPrime layer

instance (InputRecord x, Activation x) => Layer (WrappedActivation x) where
  forward layer xs = (recordInput xs layer, mapPos (const $ activation layer) xs)
  backward layer learningRate dEdY = (layer, elementwise (*) dEdY $ mapPos (const $ activationPrime layer) (getInput layer))


{-}
newtype WrappedValue a = WrappedValue a
  deriving (Show, Eq)

instance (Num a) => Num (WrappedValue a) where
  (WrappedValue a) + (WrappedValue b) = WrappedValue $ a + b
  (WrappedValue a) * (WrappedValue b) = WrappedValue $ a * b
  abs (WrappedValue a) = WrappedValue $ abs a
  signum (WrappedValue a) = WrappedValue $ signum a
  fromInteger i = WrappedValue $ fromInteger i
  negate (WrappedValue a) = WrappedValue $ negate a

instance (Fractional a) => Fractional (WrappedValue a) where
  fromRational r = WrappedValue $ fromRational r
  (WrappedValue a) / (WrappedValue b) = WrappedValue $ a / b

instance (Floating a) => Floating (WrappedValue a) where
  pi = WrappedValue pi
  exp (WrappedValue a) = WrappedValue $ exp a
  log (WrappedValue a) = WrappedValue $ log a
  sin (WrappedValue a) = WrappedValue $ sin a
  cos (WrappedValue a) = WrappedValue $ cos a
  asin (WrappedValue a) = WrappedValue $ asin a
  acos (WrappedValue a) = WrappedValue $ acos a
  atan (WrappedValue a) = WrappedValue $ atan a
  sinh (WrappedValue a) = WrappedValue $ sinh a
  cosh (WrappedValue a) = WrappedValue $ cosh a
  asinh (WrappedValue a) = WrappedValue $ asinh a
  acosh (WrappedValue a) = WrappedValue $ acosh a
  atanh (WrappedValue a) = WrappedValue $ atanh a
-}