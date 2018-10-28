// This program converts a set of digital number to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_data [FLAGS] infile DB_NAME line_num data_num
//
// where line_num denotes the amount of row, data_num denotes the amount of digital number in each row, in the format as
//   num1 num2 ... num(data_num)
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
	"The backend {lmdb, leveldb} for storing the result");

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	// Print output to stderr (while still logging)
	FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Convert a set of digital number to the leveldb/lmdb\n"
		"format used as input for Caffe.\n"
		"Usage:\n"
		"    convert_data [FLAGS] infile DB_NAME line_num data_num\n"
		);
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 5) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_data");
		return 1;
	}


	std::ifstream infile(argv[1]);
	if (!infile.is_open())
	{
		printf("Fail to open the file %s\n", argv[1]);
		return 1;
	}

	std::string dataset_names(argv[2]);

	int line_num = atoi(argv[3]);
	int data_num = atoi(argv[4]);

	// Create new DB
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(dataset_names, db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	LOG(ERROR) << "convert data";

	Datum datum;
	float t_data;
	const int kMaxKeyLength = 100;
	char key_cstr[kMaxKeyLength];
	char str[kMaxKeyLength];

	int count = 0;
	for (int line_index = 0; line_index < line_num; ++line_index) {
		datum.set_height(data_num);
		datum.set_width(1);
		datum.set_channels(1);
		datum.clear_data();
		datum.clear_float_data();
		for (int i = 0; i < data_num; ++i) {
			infile >> t_data;
			datum.add_float_data(t_data);
		}

		// sequential
		string key_str = caffe::format_int(line_index, 8);

		// Put in db
		string out;
		CHECK(datum.SerializeToString(&out));
		txn->Put(key_str, out);

		if (++count % 1000 == 0) {
			// Commit db
			txn->Commit();
			txn.reset(db->NewTransaction());
			LOG(INFO) << "Processed " << count << " lines.";
		}
		
	}
	// write the last batch
	if (count % 1000 != 0) {
		txn->Commit();
		LOG(INFO) << "Processed " << count << " lines.";
	}

	LOG(ERROR) << "Successfully convert data!";
	infile.close();

	return 0;
}
