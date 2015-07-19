(ns rnn.neuron
  (:require [rnn.math :refer [sigmoid]]))

(defrecord Unit
    [value gradient])

(defprotocol Gate
  (forward [gate] "Output value resulting from operating on the input values.")
  (backward [gate o0] "Chain the output gradient to the input gradients."))

(defrecord AddGate [u0 u1]
  Gate
  (forward
   [_]
   (Unit. (+ (:value u0) (:value u1))  0.0))
  
  (backward
   [_ o0]
   [(assoc u0 :gradient (+ (:gradient u0) (+ 1.0 (:gradient o0))))
    (assoc u1 :gradient (+ (:gradient u1) (+ 1.0 (:gradient o0))))]))

(defrecord MultiplyGate [u0 u1]
  Gate
  (forward
   [_]
   (map->Unit {:value (* (:value u0) (:value u1)) :gradient 0.0}))
  
  (backward
   [_ o0]
   [(assoc u0 :gradient (+ (:gradient u0) (* (:value u1) (:gradient o0))))
    (assoc u1 :gradient (+ (:gradient u1) (* (:value u0) (:gradient o0))))]))

(defrecord SigmoidGate [u0]
  Gate
  (forward
   [_]
   (map->Unit {:value (sigmoid (:value u0)) :gradient 0.0}))

  (backward
   [_ o0]
   (let [s (sigmoid (:value u0))]
     (assoc u0 :gradient (+ (:gradient u0) (* (* s (- 1 s)) (:gradient o0)))))))
