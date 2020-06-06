package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
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
func getCD(isFull bool, index []int) []map[string]int {
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
		for i := 0; i < len(index); i++ {
			updateDat(csvData[index[i]][1:])
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

func greedy() {
	fullCD := getCD(true, nil)
	sampleSet := mapset.NewSet()
	bestScore := initScore
	for sampleSet.Cardinality() < sampleSize {
		bestBatch := mapset.NewSet()
		curBestScore := bestScore
		for i := 0; i < batchNum; i++ {
			batch := mapset.NewSet()
			for batch.Cardinality() < batchSize {
				r := rand.Intn(dataSize)
				for sampleSet.Contains(r) {
					r = rand.Intn(dataSize)
				}
				batch.Add(r)
			}
			newSampleSet := sampleSet.Union(batch).ToSlice()
			index := make([]int, len(newSampleSet))
			for i := 0; i < len(newSampleSet); i++ {
				index[i] = newSampleSet[i].(int)
			}
			sort.Ints(index)
			sampleCD := getCD(false, index)
			if curScore := compare(fullCD, sampleCD); curScore < curBestScore {
				bestBatch = batch
				curBestScore = curScore
			}
		}
		if bestBatch.Cardinality() != 0 && bestScore/curBestScore >= 0.01 {
			sampleSet = sampleSet.Union(bestBatch)
			bestScore = curBestScore
			fmt.Println(bestScore)
		}
		fmt.Println(sampleSet.Cardinality())
	}
	index := make([]int, sampleSize)
	sampleSlice := sampleSet.ToSlice()
	for i := 0; i < sampleSize; i++ {
		index[i] = sampleSlice[i].(int)
	}
	sort.Ints(index)
	fmt.Print(compare(fullCD, getCD(false, index)))
}
func main() {
	rand.Seed(time.Now().Unix())
	parse()
	greedy()
}
