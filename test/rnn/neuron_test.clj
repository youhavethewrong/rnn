(ns rnn.neuron-test
  (:require [rnn.neuron :refer :all]
            [rnn.math :refer [double=]]
            [clojure.test :refer :all])
  (:import [rnn.neuron AddGate MultiplyGate SigmoidGate Unit]))

(deftest add-gate
  (testing "Should exercise forward and backword propagation for the add gate."
    (let [gate (AddGate.)
          forward-value (.forward gate
                                  (Unit. 2 0.0)
                                  (Unit. 3 0.0))
          backward-value (.backward gate
                                    forward-value
                                    (Unit. 2 0.0)
                                    (Unit. 3 0.0))]
      (is (= {:value 5 :gradient 0.0} forward-value))
      (is (= [(map->Unit {:value 2 :gradient 1.0})
              (map->Unit {:value 3 :gradient 1.0})] backward-value)))))

(deftest multiply-gate
  (testing "Should exercise forward and backword propagation for the multiply gate."
    (let [gate (MultiplyGate.)
          forward-value (.forward gate
                                  (Unit. 2 0.0)
                                  (Unit. 3 0.0))
          backward-value (.backward gate
                                    forward-value
                                    (Unit. 2 0.0)
                                    (Unit. 3 0.0))]
      (is (= {:value 6 :gradient 0.0} forward-value))
      (is (= [(map->Unit {:value 2 :gradient 0.0})
              (map->Unit {:value 3 :gradient 0.0})] backward-value)))))

(deftest sigmoid-gate
  (testing "Should exercise forward and backword propagation for the sig gate."
    (let [gate (SigmoidGate.)
          forward-value (.forward gate
                                  (Unit. 2 0.0)
                                  nil)
          backward-value (.backward gate
                                    forward-value
                                    (Unit. 2 0.0)
                                    nil)]
      (is (double= 0.88079 (:value forward-value) 0.001))
      (is (= 0.0 (:gradient forward-value)))
      (is (= (map->Unit {:value 2 :gradient 0.0}) backward-value)))))
