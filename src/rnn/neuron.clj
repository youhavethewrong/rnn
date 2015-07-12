(ns rnn.neuron)

(defrecord Unit
    [value gradient])

(defprotocol Gate
  (forward [gate u0 u1] "Output value resulting from operating on the input values.")
  (backward [gate o0 u0 u1] "Chain the output gradient to the input gradients."))

(defrecord MultiplyGate []
  Gate
  (forward
   [_ u0 u1]
   {:value (* (:value u0) (:value u1)) :gradient 0.0})
  
  (backward
   [_ o0 u0 u1]
   [(assoc u0 :gradient (+ (:gradient u0) (* (:value u1) (:gradient o0))))
    (assoc u1 :gradient (+ (:gradient u1) (* (:value u0) (:gradient o0))))])
)