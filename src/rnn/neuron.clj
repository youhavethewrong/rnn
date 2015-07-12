(ns rnn.neuron
  (:require [rnn.math :refer [sigmoid]]))

(defrecord Unit
    [value gradient])

(defprotocol Gate
  (forward [gate u0 u1] "Output value resulting from operating on the input values.")
  (backward [gate o0 u0 u1] "Chain the output gradient to the input gradients."))

(defrecord AddGate []
  Gate
  (forward
   [_ u0 u1]
   {:value (+ (:value u0) (:value u1)) :gradient 0.0})
  
  (backward
   [_ o0 u0 u1]
   [(assoc u0 :gradient (+ (:gradient u0) (+ 1.0 (:gradient o0))))
    (assoc u1 :gradient (+ (:gradient u1) (+ 1.0 (:gradient o0))))]))

(defrecord MultiplyGate []
  Gate
  (forward
   [_ u0 u1]
   {:value (* (:value u0) (:value u1)) :gradient 0.0})
  
  (backward
   [_ o0 u0 u1]
   [(assoc u0 :gradient (+ (:gradient u0) (* (:value u1) (:gradient o0))))
    (assoc u1 :gradient (+ (:gradient u1) (* (:value u0) (:gradient o0))))]))

(defrecord SigmoidGate []
  Gate
  (forward
   [_ u0 _]
   {:value (sigmoid (:value u0)) :gradient 0.0})

  (backward
   [_ o0 u0 _]
   (let [s (sigmoid (:value u0))]
     (assoc u0 :gradient (+ (:gradient u0) (* (* s (- 1 s)) (:gradient o0)))))))
