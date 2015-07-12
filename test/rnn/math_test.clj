(ns rnn.math-test
  (:require [rnn.math :refer :all]
            [clojure.test :refer :all]))

(deftest double=-test
  (testing "Should exercise the double= function"
    (is (double= 0.123456 0.1234567))
    (is (double= 0.123 0.1234 0.01))))

(deftest sigmoid-test
  (testing "Should exercise the sigmoid function"
    (is (double= 0.006693 (sigmoid -5) 0.0001))
    (is (double= 0.5   (sigmoid 0)) 0.0001)
    (is (double= 0.9933 (sigmoid 5)  0.0001))))


