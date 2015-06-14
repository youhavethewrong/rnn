(ns rnn.core-test
  (:require [clojure.test :refer :all]
            [rnn.core :refer :all]))

(deftest gate-tests
  (testing "Testing forward multiplier gate."
    (is (= -6 (forward-multiplier-gate -2 3))))
  (testing "Try to get something better than -6 by randomly tweaking the inputs."
    (is (> (:result (random-tweak forward-multiplier-gate -2 3))
           -6)))
  (testing "Try to get something better than -6 using derivatives"
    (is (> (:result (derivative-tweak forward-multiplier-gate -2 3))
           -6))))
