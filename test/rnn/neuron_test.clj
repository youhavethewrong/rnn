(ns rnn.neuron-test
  (:require [rnn.neuron :refer :all]
            [clojure.test :refer :all])
  (:import [rnn.neuron MultiplyGate Unit]))

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
