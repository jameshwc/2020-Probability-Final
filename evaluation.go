package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	mapset "github.com/deckarep/golang-set"
)

const (
	filename   = "t1-data.csv"
	batchNum   = 100
	batchSize  = 100
	initScore  = .08
	sampleSize = 20000
	dataSize   = 2000000
)

var csvData [][]string

func hellingerDistance(normP []float64, normQ []float64) float64 {
	s := 0.0
	for i := range normP {
		s += math.Sqrt(normP[i] * normQ[i])
	}
	return math.Sqrt(1 - s)
}
func parse() {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalf("read %s", filename)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)
	for i := 0; scanner.Scan(); i++ {
		dat := strings.Split(scanner.Text(), ",")
		csvData = append(csvData, dat)
	}
}
func getCD(isFull bool, index mapset.Set) []map[string]int {
	dat := make([]map[string]int, len(csvData[0])-1)
	for i := range dat {
		dat[i] = make(map[string]int)
	}
	updateDat := func(row []string) {
		for idx, r := range row {
			dat[idx][r]++
		}
	}
	if isFull {
		for i := range csvData {
			updateDat(csvData[i][1:])
		}
	} else {
		for index.Cardinality() != 0 {
			updateDat(csvData[index.Pop().(int)][1:])
		}
	}
	return dat
}

func compare(fullCD []map[string]int, sampleCD []map[string]int) float64 {
	var distList []float64
	for id := range fullCD {
		var normP, normQ []float64
		pSum, qSum := 0, 0
		for k := range fullCD[id] {
			pSum += fullCD[id][k]
			qSum += sampleCD[id][k]
		}
		for k := range fullCD[id] {
			normP = append(normP, float64(fullCD[id][k])/float64(pSum))
			normQ = append(normQ, float64(sampleCD[id][k])/float64(qSum))
		}
		distList = append(distList, hellingerDistance(normP, normQ))
	}
	s := 0.0
	for i := range distList {
		s += distList[i]
	}
	return s / float64(len(distList))
}

func greedy(dataSet mapset.Set) {
	fullCD := getCD(true, nil)
	sampleSet := mapset.NewSet()
	bestScore := initScore
	for sampleSet.Cardinality() < sampleSize {
		bestBatch := mapset.NewSet()
		dataSetCopy := dataSet
		curBestScore := bestScore
		for i := 0; i < batchNum; i++ {
			batch := mapset.NewSet()
			for j := 0; j < batchSize; j++ {
				batch.Add(dataSetCopy.Pop())
			}
			newSampleSet := sampleSet.Union(batch)
			sampleCD := getCD(false, newSampleSet)
			if curScore := compare(fullCD, sampleCD); curScore < curBestScore {
				bestBatch = batch
				curBestScore = curScore
			}
		}
		if bestBatch.Cardinality() != 0 && bestScore/curBestScore >= 0.01 {
			sampleSet = sampleSet.Union(bestBatch)
			dataSet = dataSet.Difference(bestBatch)
			bestScore = curBestScore
			fmt.Println(bestScore)
		}
		fmt.Println(sampleSet.Cardinality())
	}
	fmt.Print(compare(fullCD, getCD(false, sampleSet)))
}
func main() {
	rand.Seed(time.Now().Unix())
	parse()
	dataSet := mapset.NewSet()
	for i := 0; i < dataSize; i++ {
		dataSet.Add(i)
	}
	greedy(dataSet)
}
