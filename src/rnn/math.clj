(ns rnn.math
  (:require [clojure.math.numeric-tower :as math]))

(defn- scale [x y]
  (if (or (zero? x) (zero? y))
    1
    (math/abs x)))

(defn double=
  ([x y]
   (double= x y 0.00001))
  ([x y epsilon]
   (<= (math/abs (- x y))
       (* (scale x y) epsilon))))

(defn sigmoid
  "Calculates the sigmoid function for x"
  [x]
  (/ 1.0 (+ 1 (math/expt Math/E (* -1 x)))))
