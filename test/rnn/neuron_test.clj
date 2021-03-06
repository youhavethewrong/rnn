(ns rnn.neuron-test
  (:require [rnn.neuron :refer :all]
            [rnn.math :refer [double=]]
            [clojure.test :refer :all]
            [clojure.walk :as walk])
  (:import [rnn.neuron AddGate MultiplyGate SigmoidGate Unit]))

(deftest add-gate
  (testing "Should exercise forward and backward propagation for the add gate."
    (let [gate (AddGate. (Unit. 2 0.0) (Unit. 3 0.0))
          forward-value (.forward gate)
          backward-value (.backward gate forward-value)]
      (is (= (map->Unit {:value 5 :gradient 0.0}) forward-value))
      (is (= [(map->Unit {:value 2 :gradient 1.0})
              (map->Unit {:value 3 :gradient 1.0})] backward-value)))))

(deftest multiply-gate
  (testing "Should exercise forward and backward propagation for the multiply gate."
    (let [gate (MultiplyGate. (Unit. 2 0.0) (Unit. 3 0.0))
          forward-value (.forward gate)
          backward-value (.backward gate forward-value)]
      (is (= (map->Unit {:value 6 :gradient 0.0}) forward-value))
      (is (= [(map->Unit {:value 2 :gradient 0.0})
              (map->Unit {:value 3 :gradient 0.0})] backward-value)))))

(deftest sigmoid-gate
  (testing "Should exercise forward and backward propagation for the sig gate."
    (let [gate (SigmoidGate. (Unit. 2 0.0))
          forward-value (.forward gate)
          backward-value (.backward gate forward-value)]
      (is (double= 0.88079 (:value forward-value) 0.001))
      (is (= 0.0 (:gradient forward-value)))
      (is (= (map->Unit {:value 2 :gradient 0.0}) backward-value)))))

(deftest two-dimensional-neuron
  (testing "Set up a simple 2d neuron using 2 multiply gates, 2 add gates, and a sig gate."
    (let [a (Unit. 1.0 0.0)
          b (Unit. 2.0 0.0)
          c (Unit. -3.0 0.0)
          x (Unit. -1.0 0.0)
          y (Unit. 3.0 0.0)
          mulg0 (MultiplyGate. a x)
          mulg1 (MultiplyGate. b y)
          addg0 (AddGate. (.forward mulg0) (.forward mulg1))
          addg1 (AddGate. (.forward addg0) c)
          sg0 (SigmoidGate. (.forward addg1))
          fw [sg0
              [addg0
               [addg1 [mulg0 [a x]
                       mulg1 [b y]]]
               c]]
          forward-neuron (.forward sg0)
          forward-neuron (assoc forward-neuron :gradient 1.0)
          sg0 (map->SigmoidGate (.backward sg0 forward-neuron))
          addg1 (map->AddGate (.backward addg1 sg0))
          addg0 (map->AddGate (.backward addg0 addg1))
          mulg1 (map->AddGate (.backward mulg1 addg0))
          mulg0 (map->AddGate (.backward mulg0 addg0))
          ]
      (println "postwalk")
      (walk/postwalk-demo fw)
      (println "prewalk")
      (walk/prewalk-demo fw)
      (is (double= 0.8808 (:value forward-neuron) 0.0001))
      (is (double= 0.1049 (:gradient sg0) 0.001))
      (is (double= 1.1049 (:gradient addg1) 0.001))
      (is (double= 2.1049 (:gradient addg0) 0.001))
      (is (double= 4.2099 (:gradient mulg1) 0.001))
      (is (double= 2.1049 (:gradient mulg0) 0.001))
      )))
